import json
from .visitor import Visitor, visit
def leave_UnionTypeDefinition(self, node, *args):
    return 'union ' + node.name + wrap(' ', join(node.directives, ' ')) + ' = ' + join(node.types, ' | ')