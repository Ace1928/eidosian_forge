import json
from .visitor import Visitor, visit
def leave_ObjectTypeDefinition(self, node, *args):
    return join(['type', node.name, wrap('implements ', join(node.interfaces, ', ')), join(node.directives, ' '), block(node.fields)], ' ')