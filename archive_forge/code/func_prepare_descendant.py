import re
def prepare_descendant(next, token):
    token = next()
    if token[0] == '*':
        tag = '*'
    elif not token[0]:
        tag = token[1]
    else:
        raise SyntaxError('invalid descendant')

    def select(result):
        for elem in result:
            yield from elem.iterdescendants(tag)
    return select