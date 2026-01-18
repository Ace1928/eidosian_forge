import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
@_Cache.me
def feature_extra_checks(self, name):
    """
        Return a list of supported extra checks after testing them against
        the compiler.

        Parameters
        ----------
        names : str
            CPU feature name in uppercase.
        """
    assert isinstance(name, str)
    d = self.feature_supported[name]
    extra_checks = d.get('extra_checks', [])
    if not extra_checks:
        return []
    self.dist_log("Testing extra checks for feature '%s'" % name, extra_checks)
    flags = self.feature_flags(name)
    available = []
    not_available = []
    for chk in extra_checks:
        test_path = os.path.join(self.conf_check_path, 'extra_%s.c' % chk.lower())
        if not os.path.exists(test_path):
            self.dist_fatal('extra check file does not exist', test_path)
        is_supported = self.dist_test(test_path, flags + self.cc_flags['werror'])
        if is_supported:
            available.append(chk)
        else:
            not_available.append(chk)
    if not_available:
        self.dist_log('testing failed for checks', not_available, stderr=True)
    return available