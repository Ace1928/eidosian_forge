from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def adjust_list_resp(opts, resp):
    adjust_list_api_tags(opts, resp)