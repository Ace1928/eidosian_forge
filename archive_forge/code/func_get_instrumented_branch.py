from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def get_instrumented_branch(self):
    """Get a Branch object which has been instrumented"""
    self.locks = []
    b = lock_helpers.LockWrapper(self.locks, self.get_branch(), 'b')
    b.repository = lock_helpers.LockWrapper(self.locks, b.repository, 'r')
    bcf = getattr(b, 'control_files', None)
    rcf = getattr(b.repository, 'control_files', None)
    if rcf is None:
        self.combined_branch = False
    else:
        self.combined_control = bcf is rcf and bcf is not None
    try:
        b.control_files = lock_helpers.LockWrapper(self.locks, b.control_files, 'bc')
    except AttributeError:
        raise tests.TestSkipped('Could not instrument branch control files.')
    if self.combined_control:
        b.repository.control_files = lock_helpers.LockWrapper(self.locks, b.repository.control_files, 'rc')
    return b