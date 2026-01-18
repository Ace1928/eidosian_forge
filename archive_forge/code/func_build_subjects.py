from __future__ import absolute_import, division, print_function
from . import utils
def build_subjects(groups, users):
    groups_dicts = [type_name_dict('Group', g) for g in groups or []]
    users_dicts = [type_name_dict('User', u) for u in users or []]
    return groups_dicts + users_dicts