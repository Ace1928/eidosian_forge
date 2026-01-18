from __future__ import absolute_import
import re
import operator
import sys
def handle_descendants(next, token):
    """
    //...
    """
    token = next()
    if token[0] == '*':

        def iter_recursive(node):
            for name in node.child_attrs:
                for child in iterchildren(node, name):
                    yield child
                    for c in iter_recursive(child):
                        yield c
    elif not token[0]:
        node_name = token[1]

        def iter_recursive(node):
            for name in node.child_attrs:
                for child in iterchildren(node, name):
                    if type_name(child) == node_name:
                        yield child
                    for c in iter_recursive(child):
                        yield c
    else:
        raise ValueError("Expected node name after '//'")

    def select(result):
        for node in result:
            for child in iter_recursive(node):
                yield child
    return select