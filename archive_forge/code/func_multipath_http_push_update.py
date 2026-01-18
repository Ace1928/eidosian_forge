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
def multipath_http_push_update(self, update_opts):
    """
        Provides a software update via the URI specified by the
        MultipartHttpPushUri property.  Callers should adjust the 'timeout'
        variable in the base object to accommodate the size of the image and
        speed of the transfer.  For example, a 200MB image will likely take
        more than the default 10 second timeout.

        :param update_opts: The parameters for the update operation
        :return: dict containing the response of the update request
        """
    image_file = update_opts.get('update_image_file')
    targets = update_opts.get('update_targets')
    apply_time = update_opts.get('update_apply_time')
    oem_params = update_opts.get('update_oem_params')
    if not image_file:
        return {'ret': False, 'msg': 'Must specify update_image_file for the MultipartHTTPPushUpdate command'}
    if not os.path.isfile(image_file):
        return {'ret': False, 'msg': 'Must specify a valid file for the MultipartHTTPPushUpdate command'}
    try:
        with open(image_file, 'rb') as f:
            image_payload = f.read()
    except Exception as e:
        return {'ret': False, 'msg': 'Could not read file %s' % image_file}
    response = self.get_request(self.root_uri + self.update_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    if 'MultipartHttpPushUri' not in data:
        return {'ret': False, 'msg': 'Service does not support MultipartHttpPushUri'}
    update_uri = data['MultipartHttpPushUri']
    payload = {'@Redfish.OperationApplyTime': 'Immediate'}
    if targets:
        payload['Targets'] = targets
    if apply_time:
        payload['@Redfish.OperationApplyTime'] = apply_time
    if oem_params:
        payload['Oem'] = oem_params
    multipart_payload = {'UpdateParameters': {'content': json.dumps(payload), 'mime_type': 'application/json'}, 'UpdateFile': {'filename': image_file, 'content': image_payload, 'mime_type': 'application/octet-stream'}}
    response = self.post_request(self.root_uri + update_uri, multipart_payload, multipart=True)
    if response['ret'] is False:
        return response
    return {'ret': True, 'changed': True, 'msg': 'MultipartHTTPPushUpdate requested', 'update_status': self._operation_results(response['resp'], response['data'])}