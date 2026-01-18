import html.entities as htmlentitydefs
import re
import warnings
from ast import literal_eval
from collections import defaultdict
from enum import Enum
from io import StringIO
from typing import Any, NamedTuple
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file
def stringize(key, value, ignored_keys, indent, in_list=False):
    if not isinstance(key, str):
        raise NetworkXError(f'{key!r} is not a string')
    if not valid_keys.match(key):
        raise NetworkXError(f'{key!r} is not a valid key')
    if not isinstance(key, str):
        key = str(key)
    if key not in ignored_keys:
        if isinstance(value, (int, bool)):
            if key == 'label':
                yield (indent + key + ' "' + str(value) + '"')
            elif value is True:
                yield (indent + key + ' 1')
            elif value is False:
                yield (indent + key + ' 0')
            elif value < -2 ** 31 or value >= 2 ** 31:
                yield (indent + key + ' "' + str(value) + '"')
            else:
                yield (indent + key + ' ' + str(value))
        elif isinstance(value, float):
            text = repr(value).upper()
            if text == repr(float('inf')).upper():
                text = '+' + text
            else:
                epos = text.rfind('E')
                if epos != -1 and text.find('.', 0, epos) == -1:
                    text = text[:epos] + '.' + text[epos:]
            if key == 'label':
                yield (indent + key + ' "' + text + '"')
            else:
                yield (indent + key + ' ' + text)
        elif isinstance(value, dict):
            yield (indent + key + ' [')
            next_indent = indent + '  '
            for key, value in value.items():
                yield from stringize(key, value, (), next_indent)
            yield (indent + ']')
        elif isinstance(value, tuple) and key == 'label':
            yield (indent + key + f' "({','.join((repr(v) for v in value))})"')
        elif isinstance(value, (list, tuple)) and key != 'label' and (not in_list):
            if len(value) == 0:
                yield (indent + key + ' ' + f'"{value!r}"')
            if len(value) == 1:
                yield (indent + key + ' ' + f'"{LIST_START_VALUE}"')
            for val in value:
                yield from stringize(key, val, (), indent, True)
        else:
            if stringizer:
                try:
                    value = stringizer(value)
                except ValueError as err:
                    raise NetworkXError(f'{value!r} cannot be converted into a string') from err
            if not isinstance(value, str):
                raise NetworkXError(f'{value!r} is not a string')
            yield (indent + key + ' "' + escape(value) + '"')