from __future__ import absolute_import, division, print_function
from . import utils
def do_role_bindings_differ(current, desired):
    if _do_subjects_differ(current['subjects'], desired['subjects']):
        return True
    return utils.do_differ(current, desired, 'subjects')