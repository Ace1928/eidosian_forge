from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
def dump_node(self, node):
    ignored = list(node.child_attrs or []) + ['child_attrs', 'pos', 'gil_message', 'cpp_message', 'subexprs']
    values = []
    pos = getattr(node, 'pos', None)
    if pos:
        source = pos[0]
        if source:
            import os.path
            source = os.path.basename(source.get_description())
        values.append(u'%s:%s:%s' % (source, pos[1], pos[2]))
    attribute_names = dir(node)
    for attr in attribute_names:
        if attr in ignored:
            continue
        if attr.startswith('_') or attr.endswith('_'):
            continue
        try:
            value = getattr(node, attr)
        except AttributeError:
            continue
        if value is None or value == 0:
            continue
        elif isinstance(value, list):
            value = u'[...]/%d' % len(value)
        elif not isinstance(value, _PRINTABLE):
            continue
        else:
            value = repr(value)
        values.append(u'%s = %s' % (attr, value))
    return u'%s(%s)' % (node.__class__.__name__, u',\n    '.join(values))