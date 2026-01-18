from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def perform_requested_update_operations(self, update_handle):
    """
        Performs requested operations to allow the update to continue.

        :param handle: The task or job handle tracking the update
        :return: dict containing the result of the operations
        """
    update_status = self.get_update_status(update_handle)
    if update_status['ret'] is False:
        return update_status
    changed = False
    for reset in update_status['resets_requested']:
        resp = self.post_request(self.root_uri + reset['uri'], {'ResetType': reset['type']})
        if resp['ret'] is False:
            resp['changed'] = changed
            return resp
        changed = True
    msg = 'No operations required for the update'
    if changed:
        msg = 'One or more components reset to continue the update'
    return {'ret': True, 'changed': changed, 'msg': msg}