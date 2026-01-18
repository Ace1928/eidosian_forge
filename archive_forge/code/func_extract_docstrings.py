from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def extract_docstrings(module):
    docs = {}
    global_type_mode = 'complex'
    for name, obj in inspect.getmembers(module):
        if name.startswith('_'):
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            docs[name] = {}
            main_docstring = inspect.getdoc(obj) or ''
            cleaned_docstring = str.join('\n', [s for s in main_docstring.split('\n') if not re.match('^\\S+:', s)])
            docs[name]['description'] = cleaned_docstring
            docs[name]['members'] = {}
            docs['__meta__'] = {'additional_interfaces': {}}
            for member_name, member in inspect.getmembers(obj):
                if inspect.ismethod(member) or inspect.isfunction(member):
                    if member_name not in ('__init__', 'preprocess', 'postprocess'):
                        continue
                    docs[name]['members'][member_name] = {}
                    member_docstring = inspect.getdoc(member) or ''
                    type_mode = 'complex'
                    try:
                        hints = typing.get_type_hints(member)
                    except Exception:
                        type_mode = 'simple'
                        hints = member.__annotations__
                        global_type_mode = 'simple'
                    signature = inspect.signature(member)
                    for param_name, param in hints.items():
                        if param_name == 'return' and member_name == 'postprocess' or (param_name != 'return' and member_name == 'preprocess'):
                            continue
                        if type_mode == 'simple':
                            arg_names = hints.get(param_name, '')
                            additional_interfaces = {}
                            user_fn_refs = []
                        else:
                            arg_names, additional_interfaces, user_fn_refs = get_type_hints(param, module)
                        docs['__meta__']['additional_interfaces'].update(additional_interfaces)
                        docs[name]['members'][member_name][param_name] = {}
                        if param_name == 'return':
                            docstring = get_return_docstring(member_docstring)
                        else:
                            docstring = get_parameter_docstring(member_docstring, param_name)
                        add_value(docs[name]['members'][member_name][param_name], 'type', arg_names)
                        if signature.parameters.get(param_name, None) is not None:
                            default_value = signature.parameters[param_name].default
                            if default_value is not inspect._empty:
                                add_value(docs[name]['members'][member_name][param_name], 'default', format_value(default_value))
                        docs[name]['members'][member_name][param_name]['description'] = docstring
                        if member_name in ('postprocess', 'preprocess'):
                            docs[name]['members'][member_name]['value'] = find_first_non_return_key(docs[name]['members'][member_name])
                            additional_refs = get_deep(docs, ['__meta__', 'user_fn_refs', name])
                            if additional_refs is None:
                                set_deep(docs, ['__meta__', 'user_fn_refs', name], set(user_fn_refs))
                            else:
                                additional_refs = set(additional_refs)
                                additional_refs.update(user_fn_refs)
                                set_deep(docs, ['__meta__', 'user_fn_refs', name], additional_refs)
                if member_name == 'EVENTS':
                    docs[name]['events'] = {}
                    if isinstance(member, list):
                        for event in member:
                            docs[name]['events'][str(event)] = {'type': None, 'default': None, 'description': event.doc.replace('{{ component }}', name)}
        final_user_fn_refs = get_deep(docs, ['__meta__', 'user_fn_refs', name])
        if final_user_fn_refs is not None:
            set_deep(docs, ['__meta__', 'user_fn_refs', name], list(final_user_fn_refs))
    return (docs, global_type_mode)