from copy import copy
from . import ast
from .visitor_meta import QUERY_DOCUMENT_KEYS, VisitorMeta
def leave(self, node, key, parent, path, ancestors):
    result = self.visitor.leave(node, key, parent, path, ancestors)
    self.type_info.leave(node)
    return result