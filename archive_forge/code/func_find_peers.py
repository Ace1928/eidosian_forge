import re
import ast
from hacking import core
def find_peers(node):
    node_for_line = node._parent
    for _field, value in ast.iter_fields(node._parent._parent):
        if isinstance(value, list) and node_for_line in value:
            return value[value.index(node_for_line) + 1:]
        continue
    return []