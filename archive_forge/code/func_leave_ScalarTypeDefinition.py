import json
from .visitor import Visitor, visit
def leave_ScalarTypeDefinition(self, node, *args):
    return 'scalar ' + node.name + wrap(' ', join(node.directives, ' '))