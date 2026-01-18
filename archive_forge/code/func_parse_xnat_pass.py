import os
from functools import partial
def parse_xnat_pass(lines):
    empty = {'host': None, 'u': None, 'p': None}
    line = find_plus_line(lines)
    u = ('u', partial(find_token, '@'), True)
    host = ('host', partial(find_token, '='), True)
    p = ('p', partial(lambda x: (x, x)), True)

    def update_state(x, k, state):
        state[k] = x
        return state
    if line is None:
        return None
    else:
        return chain([u, host, p], line, empty, update_state)