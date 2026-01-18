from __future__ import unicode_literals
import collections
import logging
from cmakelang.lint import lintdb
def has_lint(self):
    return bool(self.get_lint())