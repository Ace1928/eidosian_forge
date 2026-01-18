import json
from .visitor import Visitor, visit
def leave_StringValue(self, node, *args):
    return json.dumps(node.value)