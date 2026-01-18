from __future__ import absolute_import, division, print_function
import json
import re
import time
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.ansible.netcommon.plugins.plugin_utils.cliconf_base import (
def edit_banner(self, candidate=None, multiline_delimiter='@', commit=True):
    """
        Edit banner on remote device
        :param banners: Banners to be loaded in json format
        :param multiline_delimiter: Line delimiter for banner
        :param commit: Boolean value that indicates if the device candidate
               configuration should be  pushed in the running configuration or discarded.
        :param diff: Boolean flag to indicate if configuration that is applied on remote host should
                     generated and returned in response or not
        :return: Returns response of executing the configuration command received
             from remote host
        """
    resp = {}
    banners_obj = json.loads(candidate)
    results = []
    requests = []
    if commit:
        for key, value in iteritems(banners_obj):
            key += ' %s' % multiline_delimiter
            self.send_command('config terminal', sendonly=True)
            for cmd in [key, value, multiline_delimiter]:
                obj = {'command': cmd, 'sendonly': True}
                results.append(self.send_command(**obj))
                requests.append(cmd)
            self.send_command('end', sendonly=True)
            time.sleep(0.1)
            results.append(self.send_command('\n'))
            requests.append('\n')
    resp['request'] = requests
    resp['response'] = results
    return resp