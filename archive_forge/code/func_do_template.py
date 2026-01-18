from __future__ import (absolute_import, division, print_function)
import ast
import datetime
import os
import pwd
import re
import time
from collections.abc import Iterator, Sequence, Mapping, MappingView, MutableMapping
from contextlib import contextmanager
from numbers import Number
from traceback import format_exc
from jinja2.exceptions import TemplateSyntaxError, UndefinedError, SecurityError
from jinja2.loaders import FileSystemLoader
from jinja2.nativetypes import NativeEnvironment
from jinja2.runtime import Context, StrictUndefined
from ansible import constants as C
from ansible.errors import (
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.common.collections import is_sequence
from ansible.plugins.loader import filter_loader, lookup_loader, test_loader
from ansible.template.native_helpers import ansible_native_concat, ansible_eval_concat, ansible_concat
from ansible.template.template import AnsibleJ2Template
from ansible.template.vars import AnsibleJ2Vars
from ansible.utils.display import Display
from ansible.utils.listify import listify_lookup_plugin_terms
from ansible.utils.native_jinja import NativeJinjaText
from ansible.utils.unsafe_proxy import to_unsafe_text, wrap_var, AnsibleUnsafeText, AnsibleUnsafeBytes, NativeJinjaUnsafeText
def do_template(self, data, preserve_trailing_newlines=True, escape_backslashes=True, fail_on_undefined=None, overrides=None, disable_lookups=False, convert_data=False):
    if self.jinja2_native and (not isinstance(data, string_types)):
        return data
    data_newlines = _count_newlines_from_end(data)
    if fail_on_undefined is None:
        fail_on_undefined = self._fail_on_undefined_errors
    try:
        data, myenv = _create_overlay(data, overrides, self.environment)
        self._compile_single_var(myenv)
        if escape_backslashes:
            data = _escape_backslashes(data, myenv)
        try:
            t = myenv.from_string(data)
        except (TemplateSyntaxError, SyntaxError) as e:
            raise AnsibleError('template error while templating string: %s. String: %s' % (to_native(e), to_native(data)), orig_exc=e)
        except Exception as e:
            if 'recursion' in to_native(e):
                raise AnsibleError('recursive loop detected in template string: %s' % to_native(data), orig_exc=e)
            else:
                return data
        if disable_lookups:
            t.globals['query'] = t.globals['q'] = t.globals['lookup'] = self._fail_lookup
        jvars = AnsibleJ2Vars(self, t.globals)
        cached_context = self.cur_context
        myenv.concat = myenv.__class__.concat
        if not self.jinja2_native and (not convert_data):
            myenv.concat = ansible_concat
        self.cur_context = t.new_context(jvars, shared=True)
        rf = t.root_render_func(self.cur_context)
        try:
            res = myenv.concat(rf)
            unsafe = getattr(self.cur_context, 'unsafe', False)
            if unsafe:
                res = wrap_var(res)
        except TypeError as te:
            if 'AnsibleUndefined' in to_native(te):
                errmsg = 'Unable to look up a name or access an attribute in template string (%s).\n' % to_native(data)
                errmsg += "Make sure your variable name does not contain invalid characters like '-': %s" % to_native(te)
                raise AnsibleUndefinedVariable(errmsg, orig_exc=te)
            else:
                display.debug('failing because of a type error, template data is: %s' % to_text(data))
                raise AnsibleError('Unexpected templating type error occurred on (%s): %s' % (to_native(data), to_native(te)), orig_exc=te)
        finally:
            self.cur_context = cached_context
        if isinstance(res, string_types) and preserve_trailing_newlines:
            res_newlines = _count_newlines_from_end(res)
            if data_newlines > res_newlines:
                res += myenv.newline_sequence * (data_newlines - res_newlines)
                if unsafe:
                    res = wrap_var(res)
        return res
    except (UndefinedError, AnsibleUndefinedVariable) as e:
        if fail_on_undefined:
            raise AnsibleUndefinedVariable(e, orig_exc=e)
        else:
            display.debug('Ignoring undefined failure: %s' % to_text(e))
            return data