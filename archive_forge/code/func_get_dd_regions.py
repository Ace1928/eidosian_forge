from __future__ import (absolute_import, division, print_function)
import os
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib  # noqa: F401, pylint: disable=unused-import
from ansible.module_utils.six.moves import configparser
from os.path import expanduser
from uuid import UUID
def get_dd_regions():
    """
    Get the list of available regions whose vendor is Dimension Data.
    """
    all_regions = API_ENDPOINTS.keys()
    regions = [region[3:] for region in all_regions if region.startswith('dd-')]
    return regions