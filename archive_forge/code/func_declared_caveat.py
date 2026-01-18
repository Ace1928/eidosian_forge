import collections
import pyrfc3339
from ._conditions import (
def declared_caveat(key, value):
    """Returns a "declared" caveat asserting that the given key is
    set to the given value.

    If a macaroon has exactly one first party caveat asserting the value of a
    particular key, then infer_declared will be able to infer the value, and
    then the check will allow the declared value if it has the value
    specified here.

    If the key is empty or contains a space, it will return an error caveat.
    """
    if key.find(' ') >= 0 or key == '':
        return error_caveat('invalid caveat \'declared\' key "{}"'.format(key))
    return _first_party(COND_DECLARED, key + ' ' + value)