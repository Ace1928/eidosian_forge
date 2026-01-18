from __future__ import absolute_import
import cython
from io import StringIO
import re
import sys
from unicodedata import lookup as lookup_unicodechar, category as unicode_category
from functools import partial, reduce
from .Scanning import PyrexScanner, FileSourceDescriptor, tentatively_scan
from . import Nodes
from . import ExprNodes
from . import Builtin
from . import StringEncoding
from .StringEncoding import EncodedString, bytes_literal, _unicode, _bytes
from .ModuleNode import ModuleNode
from .Errors import error, warning
from .. import Utils
from . import Future
from . import Options
def p_dict_or_set_maker(s):
    pos = s.position()
    s.next()
    if s.sy == '}':
        s.next()
        return ExprNodes.DictNode(pos, key_value_pairs=[])
    parts = []
    target_type = 0
    last_was_simple_item = False
    while True:
        if s.sy in ('*', '**'):
            if target_type == 0:
                target_type = 1 if s.sy == '*' else 2
            elif target_type != len(s.sy):
                s.error('unexpected %sitem found in %s literal' % (s.sy, 'set' if target_type == 1 else 'dict'))
            s.next()
            if s.sy == '*':
                s.error("expected expression, found '*'")
            item = p_starred_expr(s)
            parts.append(item)
            last_was_simple_item = False
        else:
            item = p_test(s)
            if target_type == 0:
                target_type = 2 if s.sy == ':' else 1
            if target_type == 2:
                s.expect(':')
                key = item
                value = p_test(s)
                item = ExprNodes.DictItemNode(key.pos, key=key, value=value)
            if last_was_simple_item:
                parts[-1].append(item)
            else:
                parts.append([item])
                last_was_simple_item = True
        if s.sy == ',':
            s.next()
            if s.sy == '}':
                break
        else:
            break
    if s.sy in ('for', 'async'):
        if len(parts) == 1 and isinstance(parts[0], list) and (len(parts[0]) == 1):
            item = parts[0][0]
            if target_type == 2:
                assert isinstance(item, ExprNodes.DictItemNode), type(item)
                comprehension_type = Builtin.dict_type
                append = ExprNodes.DictComprehensionAppendNode(item.pos, key_expr=item.key, value_expr=item.value)
            else:
                comprehension_type = Builtin.set_type
                append = ExprNodes.ComprehensionAppendNode(item.pos, expr=item)
            loop = p_comp_for(s, append)
            s.expect('}')
            return ExprNodes.ComprehensionNode(pos, loop=loop, append=append, type=comprehension_type)
        else:
            if len(parts) == 1 and (not isinstance(parts[0], list)):
                s.error('iterable unpacking cannot be used in comprehension')
            else:
                s.expect('}')
            return ExprNodes.DictNode(pos, key_value_pairs=[])
    s.expect('}')
    if target_type == 1:
        items = []
        set_items = []
        for part in parts:
            if isinstance(part, list):
                set_items.extend(part)
            else:
                if set_items:
                    items.append(ExprNodes.SetNode(set_items[0].pos, args=set_items))
                    set_items = []
                items.append(part)
        if set_items:
            items.append(ExprNodes.SetNode(set_items[0].pos, args=set_items))
        if len(items) == 1 and items[0].is_set_literal:
            return items[0]
        return ExprNodes.MergedSequenceNode(pos, args=items, type=Builtin.set_type)
    else:
        items = []
        dict_items = []
        for part in parts:
            if isinstance(part, list):
                dict_items.extend(part)
            else:
                if dict_items:
                    items.append(ExprNodes.DictNode(dict_items[0].pos, key_value_pairs=dict_items))
                    dict_items = []
                items.append(part)
        if dict_items:
            items.append(ExprNodes.DictNode(dict_items[0].pos, key_value_pairs=dict_items))
        if len(items) == 1 and items[0].is_dict_literal:
            return items[0]
        return ExprNodes.MergedDictNode(pos, keyword_args=items, reject_duplicates=False)