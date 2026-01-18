import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
@_Cache.me
def feature_test(self, name, force_flags=None, macros=[]):
    """
        Test a certain CPU feature against the compiler through its own
        check file.

        Parameters
        ----------
        name : str
            Supported CPU feature name.

        force_flags : list or None, optional
            If None(default), the returned flags from `feature_flags()`
            will be used.

        macros : list of tuples, optional
            A list of C macro definitions.
        """
    if force_flags is None:
        force_flags = self.feature_flags(name)
    self.dist_log("testing feature '%s' with flags (%s)" % (name, ' '.join(force_flags)))
    test_path = os.path.join(self.conf_check_path, 'cpu_%s.c' % name.lower())
    if not os.path.exists(test_path):
        self.dist_fatal('feature test file is not exist', test_path)
    test = self.dist_test(test_path, force_flags + self.cc_flags['werror'], macros=macros)
    if not test:
        self.dist_log('testing failed', stderr=True)
    return test