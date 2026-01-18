from __future__ import (absolute_import, division, print_function)
def format_ttl(ttl):
    if ttl is None:
        return 'default'
    sec = ttl % 60
    ttl //= 60
    min = ttl % 60
    ttl //= 60
    h = ttl
    result = []
    if h:
        result.append('{0}h'.format(h))
    if min:
        result.append('{0}m'.format(min))
    if sec:
        result.append('{0}s'.format(sec))
    return ' '.join(result)