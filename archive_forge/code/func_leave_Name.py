import json
from .visitor import Visitor, visit
def leave_Name(self, node, *args):
    return node.value