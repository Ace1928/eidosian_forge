import json
from .visitor import Visitor, visit
def leave_InlineFragment(self, node, *args):
    return join(['...', wrap('on ', node.type_condition), join(node.directives, ''), node.selection_set], ' ')