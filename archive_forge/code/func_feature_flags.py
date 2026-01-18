import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
@_Cache.me
def feature_flags(self, names):
    """
        Return a list of CPU features flags sorted from the lowest
        to highest interest.
        """
    names = self.feature_sorted(self.feature_implies_c(names))
    flags = []
    for n in names:
        d = self.feature_supported[n]
        f = d.get('flags', [])
        if not f or not self.cc_test_flags(f):
            continue
        flags += f
    return self.cc_normalize_flags(flags)