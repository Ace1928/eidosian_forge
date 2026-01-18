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
def get_update_status(self, update_handle):
    """
        Gets the status of an update operation.

        :param handle: The task or job handle tracking the update
        :return: dict containing the response of the update status
        """
    if not update_handle:
        return {'ret': False, 'msg': 'Must provide a handle tracking the update.'}
    response = self.get_request(self.root_uri + update_handle, allow_no_resp=True)
    if response['ret'] is False:
        return response
    return self._operation_results(response['resp'], response['data'], update_handle)