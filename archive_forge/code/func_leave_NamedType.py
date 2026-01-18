import json
from .visitor import Visitor, visit
def leave_NamedType(self, node, *args):
    return node.name