from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
def full_tag_name(self, tag):
    """ Returns the full tag name in manageiq
        """
    return '/managed/{tag_category}/{tag_name}'.format(tag_category=tag['category'], tag_name=tag['name'])