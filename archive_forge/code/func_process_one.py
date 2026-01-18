import inspect
import itertools
import re
import typing as T
from textwrap import dedent
from .common import (
def process_one(one: T.Union[DocstringParam, DocstringReturns, DocstringRaises]):
    if isinstance(one, DocstringParam):
        head = one.arg_name
    elif isinstance(one, DocstringReturns):
        head = one.return_name
    else:
        head = None
    if one.type_name and head:
        head += f' : {one.type_name}'
    elif one.type_name:
        head = one.type_name
    elif not head:
        head = ''
    if isinstance(one, DocstringParam) and one.is_optional:
        head += ', optional'
    if one.description:
        body = f'\n{indent}'.join([head] + one.description.splitlines())
        parts.append(body)
    else:
        parts.append(head)