import builtins
import contextlib
import enum
import inspect
import re
import sys
import types
import typing
from functools import partial, partialmethod
from importlib import import_module
from inspect import Parameter, isclass, ismethod, ismethoddescriptor, ismodule  # NOQA
from io import StringIO
from types import MethodType, ModuleType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, cast
from sphinx.pycode.ast import ast  # for py36-37
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging
from sphinx.util.typing import ForwardRef
from sphinx.util.typing import stringify as stringify_annotation
def stringify_signature(sig: inspect.Signature, show_annotation: bool=True, show_return_annotation: bool=True, unqualified_typehints: bool=False) -> str:
    """Stringify a Signature object.

    :param show_annotation: If enabled, show annotations on the signature
    :param show_return_annotation: If enabled, show annotation of the return value
    :param unqualified_typehints: If enabled, show annotations as unqualified
                                  (ex. io.StringIO -> StringIO)
    """
    if unqualified_typehints:
        mode = 'smart'
    else:
        mode = 'fully-qualified'
    args = []
    last_kind = None
    for param in sig.parameters.values():
        if param.kind != param.POSITIONAL_ONLY and last_kind == param.POSITIONAL_ONLY:
            args.append('/')
        if param.kind == param.KEYWORD_ONLY and last_kind in (param.POSITIONAL_OR_KEYWORD, param.POSITIONAL_ONLY, None):
            args.append('*')
        arg = StringIO()
        if param.kind == param.VAR_POSITIONAL:
            arg.write('*' + param.name)
        elif param.kind == param.VAR_KEYWORD:
            arg.write('**' + param.name)
        else:
            arg.write(param.name)
        if show_annotation and param.annotation is not param.empty:
            arg.write(': ')
            arg.write(stringify_annotation(param.annotation, mode))
        if param.default is not param.empty:
            if show_annotation and param.annotation is not param.empty:
                arg.write(' = ')
            else:
                arg.write('=')
            arg.write(object_description(param.default))
        args.append(arg.getvalue())
        last_kind = param.kind
    if last_kind == Parameter.POSITIONAL_ONLY:
        args.append('/')
    if sig.return_annotation is Parameter.empty or show_annotation is False or show_return_annotation is False:
        return '(%s)' % ', '.join(args)
    else:
        annotation = stringify_annotation(sig.return_annotation, mode)
        return '(%s) -> %s' % (', '.join(args), annotation)