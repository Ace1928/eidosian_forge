from __future__ import (absolute_import, division, print_function)
import csv
import os
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination, strip_substr_dict
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
from datetime import datetime
def get_message_payload(module):
    mlist = []
    if module.params.get('message_file'):
        csvpath = module.params.get('message_file')
        if not os.path.isfile(csvpath):
            module.exit_json(failed=True, msg=CSV_PATH.format(csvpath))
        with open(csvpath) as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                mlist.extend(row)
            if mlist[0].lower().startswith('message'):
                mlist.pop(0)
    elif module.params.get('message_ids'):
        mlist = module.params.get('message_ids')
    return mlist