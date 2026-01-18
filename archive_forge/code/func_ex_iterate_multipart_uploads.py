import os
import hmac
import time
import base64
from typing import Dict, Optional
from hashlib import sha1
from datetime import datetime
import libcloud.utils.py3
from libcloud.utils.py3 import b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import findtext, fixxpath
from libcloud.common.aws import (
from libcloud.common.base import RawResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def ex_iterate_multipart_uploads(self, container, prefix=None, delimiter=None):
    """
        Extension method for listing all in-progress S3 multipart uploads.

        Each multipart upload which has not been committed or aborted is
        considered in-progress.

        :param container: The container holding the uploads
        :type container: :class:`Container`

        :keyword prefix: Print only uploads of objects with this prefix
        :type prefix: ``str``

        :keyword delimiter: The object/key names are grouped based on
            being split by this delimiter
        :type delimiter: ``str``

        :return: A generator of S3MultipartUpload instances.
        :rtype: ``generator`` of :class:`S3MultipartUpload`
        """
    if not self.supports_s3_multipart_upload:
        raise LibcloudError('Feature not supported', driver=self)
    request_path = self._get_container_path(container)
    params = {'max-uploads': RESPONSES_PER_REQUEST, 'uploads': ''}
    if prefix:
        params['prefix'] = prefix
    if delimiter:
        params['delimiter'] = delimiter

    def finder(node, text):
        return node.findtext(fixxpath(xpath=text, namespace=self.namespace))
    while True:
        response = self.connection.request(request_path, params=params)
        if response.status != httplib.OK:
            raise LibcloudError('Error fetching multipart uploads. Got code: %s' % response.status, driver=self)
        body = response.parse_body()
        for node in body.findall(fixxpath(xpath='Upload', namespace=self.namespace)):
            initiator = node.find(fixxpath(xpath='Initiator', namespace=self.namespace))
            owner = node.find(fixxpath(xpath='Owner', namespace=self.namespace))
            key = finder(node, 'Key')
            upload_id = finder(node, 'UploadId')
            created_at = finder(node, 'Initiated')
            initiator = finder(initiator, 'DisplayName')
            owner = finder(owner, 'DisplayName')
            yield S3MultipartUpload(key, upload_id, created_at, initiator, owner)
        is_truncated = body.findtext(fixxpath(xpath='IsTruncated', namespace=self.namespace))
        if is_truncated.lower() == 'false':
            break
        upload_marker = body.findtext(fixxpath(xpath='NextUploadIdMarker', namespace=self.namespace))
        key_marker = body.findtext(fixxpath(xpath='NextKeyMarker', namespace=self.namespace))
        params['key-marker'] = key_marker
        params['upload-id-marker'] = upload_marker