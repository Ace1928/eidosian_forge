from __future__ import annotations
import collections
import enum
import functools
import hashlib
import inspect
import io
import os
import pickle
import sys
import tempfile
import textwrap
import threading
import weakref
from typing import Any, Callable, Dict, Pattern, Type, Union
from streamlit import config, file_util, type_util, util
from streamlit.errors import MarkdownFormattedException, StreamlitAPIException
from streamlit.folder_black_list import FolderBlackList
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit.util import HASHLIB_KWARGS
from, try looking at the hash chain below for an object that you do recognize,
from, try looking at the hash chain below for an object that you do recognize,
def get_referenced_objects(code, context: Context) -> list[Any]:
    tos: Any = None
    lineno = None
    refs: list[Any] = []

    def set_tos(t):
        nonlocal tos
        if tos is not None:
            refs.append(tos)
        tos = t
    import dis
    for op in dis.get_instructions(code):
        try:
            if op.starts_line is not None:
                lineno = op.starts_line
            if op.opname in ['LOAD_GLOBAL', 'LOAD_NAME']:
                if op.argval in context.globals:
                    set_tos(context.globals[op.argval])
                else:
                    set_tos(op.argval)
            elif op.opname in ['LOAD_DEREF', 'LOAD_CLOSURE']:
                set_tos(context.cells.values[op.argval])
            elif op.opname == 'IMPORT_NAME':
                try:
                    import importlib
                    set_tos(importlib.import_module(op.argval))
                except ImportError:
                    set_tos(op.argval)
            elif op.opname in ['LOAD_METHOD', 'LOAD_ATTR', 'IMPORT_FROM']:
                if tos is None:
                    refs.append(op.argval)
                elif isinstance(tos, str):
                    tos += '.' + op.argval
                else:
                    tos = getattr(tos, op.argval)
            elif op.opname == 'DELETE_FAST' and tos:
                del context.varnames[op.argval]
                tos = None
            elif op.opname == 'STORE_FAST' and tos:
                context.varnames[op.argval] = tos
                tos = None
            elif op.opname == 'LOAD_FAST' and op.argval in context.varnames:
                set_tos(context.varnames[op.argval])
            elif tos is not None:
                refs.append(tos)
                tos = None
        except Exception as e:
            raise UserHashError(e, code, lineno=lineno)
    return refs