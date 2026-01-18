import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
@_Cache.me
def feature_is_supported(self, name, force_flags=None, macros=[]):
    """
        Check if a certain CPU feature is supported by the platform and compiler.

        Parameters
        ----------
        name : str
            CPU feature name in uppercase.

        force_flags : list or None, optional
            If None(default), default compiler flags for every CPU feature will
            be used during test.

        macros : list of tuples, optional
            A list of C macro definitions.
        """
    assert name.isupper()
    assert force_flags is None or isinstance(force_flags, list)
    supported = name in self.feature_supported
    if supported:
        for impl in self.feature_implies(name):
            if not self.feature_test(impl, force_flags, macros=macros):
                return False
        if not self.feature_test(name, force_flags, macros=macros):
            return False
    return supported