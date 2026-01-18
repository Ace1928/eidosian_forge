from __future__ import absolute_import
import re
import operator
import sys
def handle_name(next, token):
    """
    /NodeName/
    or
    func(...)
    """
    name = token[1]
    if name in functions:
        return functions[name](next, token)

    def select(result):
        for node in result:
            for attr_name in node.child_attrs:
                for child in iterchildren(node, attr_name):
                    if type_name(child) == name:
                        yield child
    return select