import json
from .visitor import Visitor, visit
def leave_InterfaceTypeDefinition(self, node, *args):
    return 'interface ' + node.name + wrap(' ', join(node.directives, ' ')) + ' ' + block(node.fields)