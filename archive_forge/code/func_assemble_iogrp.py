from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def assemble_iogrp(self):
    if self.iogrp:
        temp = []
        invalid = []
        active_iogrp = []
        existing_iogrp = []
        if self.iogrp:
            existing_iogrp = [item.strip() for item in self.iogrp.split(',') if item]
        uni_exi_iogrp = set(existing_iogrp)
        if len(existing_iogrp) != len(uni_exi_iogrp):
            self.module.fail_json(msg='Duplicate iogrp detected.')
        active_iogrp = [item['name'] for item in self.restapi.svc_obj_info('lsiogrp', None, None) if int(item['node_count']) > 0]
        for item in existing_iogrp:
            item = item.strip()
            if item not in active_iogrp:
                invalid.append(item)
            else:
                temp.append(item)
        if invalid:
            self.module.fail_json(msg='Empty or non-existing iogrp detected: %s' % invalid)
        self.iogrp = temp