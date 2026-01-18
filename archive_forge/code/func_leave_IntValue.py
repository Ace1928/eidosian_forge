import json
from .visitor import Visitor, visit
def leave_IntValue(self, node, *args):
    return node.value