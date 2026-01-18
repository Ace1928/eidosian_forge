import json
from .visitor import Visitor, visit
def leave_EnumValue(self, node, *args):
    return node.value