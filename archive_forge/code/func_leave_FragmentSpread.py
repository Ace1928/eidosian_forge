import json
from .visitor import Visitor, visit
def leave_FragmentSpread(self, node, *args):
    return '...' + node.name + wrap(' ', join(node.directives, ' '))