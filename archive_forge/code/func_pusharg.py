from __future__ import print_function
from __future__ import unicode_literals
import contextlib
from cmakelang import common
@contextlib.contextmanager
def pusharg(self, node):
    self.argstack.append(node)
    yield None
    if not self.argstack:
        raise common.InternalError('Unexpected empty argstack, expected {}'.format(node))
    if self.argstack[-1] is not node:
        raise common.InternalError('Unexpected node {} on argstack, expecting {}'.format(self.argstack[-1], node))
    self.argstack.pop(-1)