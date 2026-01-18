import re
def prepare_star(next, token):

    def select(result):
        for elem in result:
            yield from elem.iterchildren('*')
    return select