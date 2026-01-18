import re
def prepare_child(next, token):
    tag = token[1]

    def select(result):
        for elem in result:
            yield from elem.iterchildren(tag)
    return select