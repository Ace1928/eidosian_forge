import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def feature_names(self, names=None, force_flags=None, macros=[]):
    """
        Returns a set of CPU feature names that supported by platform and the **C** compiler.

        Parameters
        ----------
        names : sequence or None, optional
            Specify certain CPU features to test it against the **C** compiler.
            if None(default), it will test all current supported features.
            **Note**: feature names must be in upper-case.

        force_flags : list or None, optional
            If None(default), default compiler flags for every CPU feature will
            be used during the test.

        macros : list of tuples, optional
            A list of C macro definitions.
        """
    assert names is None or (not isinstance(names, str) and hasattr(names, '__iter__'))
    assert force_flags is None or isinstance(force_flags, list)
    if names is None:
        names = self.feature_supported.keys()
    supported_names = set()
    for f in names:
        if self.feature_is_supported(f, force_flags=force_flags, macros=macros):
            supported_names.add(f)
    return supported_names