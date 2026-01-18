from __future__ import absolute_import
import re
import operator
import sys
def handle_predicate(next, token):
    token = next()
    selector = []
    while token[0] != ']':
        selector.append(operations[token[0]](next, token))
        try:
            token = next()
        except StopIteration:
            break
        else:
            if token[0] == '/':
                token = next()
        if not token[0] and token[1] == 'and':
            return logical_and(selector, handle_predicate(next, token))

    def select(result):
        for node in result:
            subresult = iter((node,))
            for select in selector:
                subresult = select(subresult)
            predicate_result = _get_first_or_none(subresult)
            if predicate_result is not None:
                yield node
    return select