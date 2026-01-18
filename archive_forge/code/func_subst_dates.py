import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def subst_dates(string):
    """Replace date strings with constant values."""
    return re.sub('\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2} [-\\+]\\d{4}', 'YYYY-MM-DD HH:MM:SS +ZZZZ', string)