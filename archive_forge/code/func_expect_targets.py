import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def expect_targets(self, targets, groups={}, **kwargs):
    match = self.arg_regex(**kwargs)
    if match is None:
        return
    targets, _ = self.get_targets(targets=targets, groups=groups, **kwargs)
    targets = ' '.join(targets)
    if not match:
        if len(targets) != 0:
            raise AssertionError('expected empty targets, not "%s"' % targets)
        return
    if not re.match(match, targets, re.IGNORECASE):
        raise AssertionError('targets "%s" not match "%s"' % (targets, match))