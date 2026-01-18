import json
from .visitor import Visitor, visit
def leave_FieldDefinition(self, node, *args):
    return node.name + wrap('(', join(node.arguments, ', '), ')') + ': ' + node.type + wrap(' ', join(node.directives, ' '))