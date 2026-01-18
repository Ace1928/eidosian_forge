from __future__ import absolute_import
import re
import operator
import sys
def handle_attribute(next, token):
    token = next()
    if token[0]:
        raise ValueError('Expected attribute name')
    name = token[1]
    value = None
    try:
        token = next()
    except StopIteration:
        pass
    else:
        if token[0] == '=':
            value = parse_path_value(next)
    readattr = operator.attrgetter(name)
    if value is None:

        def select(result):
            for node in result:
                try:
                    attr_value = readattr(node)
                except AttributeError:
                    continue
                if attr_value is not None:
                    yield attr_value
    else:

        def select(result):
            for node in result:
                try:
                    attr_value = readattr(node)
                except AttributeError:
                    continue
                if attr_value == value:
                    yield attr_value
                elif isinstance(attr_value, bytes) and isinstance(value, _unicode) and (attr_value == value.encode()):
                    yield attr_value
    return select