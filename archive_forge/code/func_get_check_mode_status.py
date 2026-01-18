from __future__ import (absolute_import, division, print_function)
import os
import json
import time
from ssl import SSLError
from xml.etree import ElementTree as ET
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_check_mode_status(status, module):
    if status['job_details']['Data']['GetRepoBasedUpdateList_OUTPUT'].get('Message') == MESSAGE.rstrip('.') and status.get('JobStatus') == 'Completed':
        if module.check_mode:
            module.exit_json(msg='No changes found to commit!')
        module.exit_json(msg=EXIT_MESSAGE)