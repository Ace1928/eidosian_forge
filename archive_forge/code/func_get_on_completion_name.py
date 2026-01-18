import re
from collections import namedtuple
from textwrap import dedent
from itertools import chain
from functools import wraps
from inspect import Parameter
from parso.python.parser import Parser
from parso.python import tree
from jedi.inference.base_value import NO_VALUES
from jedi.inference.syntax_tree import infer_atom
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.compiled import get_string_value_set
from jedi.cache import signature_time_cache, memoize_method
from jedi.parser_utils import get_parent_scope
def get_on_completion_name(module_node, lines, position):
    leaf = module_node.get_leaf_for_position(position)
    if leaf is None or leaf.type in ('string', 'error_leaf'):
        line = lines[position[0] - 1]
        return re.search('(?!\\d)\\w+$|$', line[:position[1]]).group(0)
    elif leaf.type not in ('name', 'keyword'):
        return ''
    return leaf.value[:position[1] - leaf.start_pos[1]]