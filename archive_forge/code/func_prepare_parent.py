import re
def prepare_parent(next, token):

    def select(result):
        for elem in result:
            parent = elem.getparent()
            if parent is not None:
                yield parent
    return select