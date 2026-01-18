from ... import errors, tests, transport
from .. import index as _mod_index
def make_value(self, number):
    return b'%d' % number + b'Y' * 100