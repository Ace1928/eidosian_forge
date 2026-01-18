import json
from .visitor import Visitor, visit
def leave_Variable(self, node, *args):
    return '$' + node.name