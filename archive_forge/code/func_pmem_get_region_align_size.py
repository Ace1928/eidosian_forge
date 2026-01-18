from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def pmem_get_region_align_size(self, region):
    aligns = []
    for rg in region:
        if rg['align'] not in aligns:
            aligns.append(rg['align'])
    return aligns