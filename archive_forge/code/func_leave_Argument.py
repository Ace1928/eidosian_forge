import json
from .visitor import Visitor, visit
def leave_Argument(self, node, *args):
    return node.name + ': ' + node.value