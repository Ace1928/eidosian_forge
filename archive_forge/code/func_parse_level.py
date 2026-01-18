import os
import re
import sys
import traceback
import ast
import importlib
from re import sub, findall
from types import CodeType
from functools import partial
from collections import OrderedDict, defaultdict
import kivy.lang.builder  # imported as absolute to avoid circular import
from kivy.logger import Logger
from kivy.cache import Cache
from kivy import require
from kivy.resources import resource_find
from kivy.utils import rgba
import kivy.metrics as Metrics
def parse_level(self, level, lines, spaces=0):
    """Parse the current level (level * spaces) indentation.
        """
    indent = spaces * level if spaces > 0 else 0
    objects = []
    current_object = None
    current_property = None
    current_propobject = None
    i = 0
    while i < len(lines):
        line = lines[i]
        ln, content = line
        tmp = content.lstrip(' \t')
        tmp = content[:len(content) - len(tmp)]
        tmp = tmp.replace('\t', '    ')
        if spaces == 0:
            spaces = len(tmp)
        count = len(tmp)
        if spaces > 0 and count % spaces != 0:
            raise ParserException(self, ln, 'Invalid indentation, must be a multiple of %s spaces' % spaces)
        content = content.strip()
        rlevel = count // spaces if spaces > 0 else 0
        if count < indent:
            return (objects, lines[i - 1:])
        elif count == indent:
            x = content.split(':', 1)
            if not x[0]:
                raise ParserException(self, ln, 'Identifier missing')
            if len(x) == 2 and len(x[1]) and (not x[1].lstrip().startswith('#')):
                raise ParserException(self, ln, 'Invalid data after declaration')
            name = x[0].rstrip()
            if count != 0:
                if False in [ord(z) in Parser.PROP_RANGE for z in name]:
                    raise ParserException(self, ln, 'Invalid class name')
            current_object = ParserRule(self, ln, name, rlevel)
            current_property = None
            objects.append(current_object)
        elif count == indent + spaces:
            x = content.split(':', 1)
            if not x[0]:
                raise ParserException(self, ln, 'Identifier missing')
            current_property = None
            name = x[0].rstrip()
            ignore_prev = name[0] == '-'
            if ignore_prev:
                name = name[1:]
            if ord(name[0]) in Parser.CLASS_RANGE:
                if ignore_prev:
                    raise ParserException(self, ln, 'clear previous, `-`, not allowed here')
                _objects, _lines = self.parse_level(level + 1, lines[i:], spaces)
                if current_object is None:
                    raise ParserException(self, ln, 'Invalid indentation')
                current_object.children = _objects
                lines = _lines
                i = 0
            else:
                if name not in Parser.PROP_ALLOWED:
                    if not all((ord(z) in Parser.PROP_RANGE for z in name)):
                        raise ParserException(self, ln, 'Invalid property name')
                if len(x) == 1:
                    raise ParserException(self, ln, 'Syntax error')
                value = x[1].strip()
                if name == 'id':
                    if len(value) <= 0:
                        raise ParserException(self, ln, 'Empty id')
                    if value in ('self', 'root'):
                        raise ParserException(self, ln, 'Invalid id, cannot be "self" or "root"')
                    current_object.id = value
                elif len(value):
                    rule = ParserRuleProperty(self, ln, name, value, ignore_prev)
                    if name[:3] == 'on_':
                        current_object.handlers.append(rule)
                    else:
                        ignore_prev = False
                        current_object.properties[name] = rule
                else:
                    current_property = name
                    current_propobject = None
                if ignore_prev:
                    raise ParserException(self, ln, 'clear previous, `-`, not allowed here')
        elif count == indent + 2 * spaces:
            if current_property in ('canvas', 'canvas.after', 'canvas.before'):
                _objects, _lines = self.parse_level(level + 2, lines[i:], spaces)
                rl = ParserRule(self, ln, current_property, rlevel)
                rl.children = _objects
                if current_property == 'canvas':
                    current_object.canvas_root = rl
                elif current_property == 'canvas.before':
                    current_object.canvas_before = rl
                else:
                    current_object.canvas_after = rl
                current_property = None
                lines = _lines
                i = 0
            elif current_propobject is None:
                current_propobject = ParserRuleProperty(self, ln, current_property, content)
                if not current_property:
                    raise ParserException(self, ln, 'Invalid indentation')
                if current_property[:3] == 'on_':
                    current_object.handlers.append(current_propobject)
                else:
                    current_object.properties[current_property] = current_propobject
            else:
                current_propobject.value += '\n' + content
        else:
            raise ParserException(self, ln, 'Invalid indentation (too many levels)')
        i += 1
    return (objects, [])