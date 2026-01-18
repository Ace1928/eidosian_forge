import os
import boto.glacier
from boto.compat import json
from boto.connection import AWSAuthConnection
from boto.glacier.exceptions import UnexpectedHTTPResponseError
from boto.glacier.response import GlacierResponse
from boto.glacier.utils import ResettingFileSender
def list_parts(self, vault_name, upload_id, limit=None, marker=None):
    """
        This operation lists the parts of an archive that have been
        uploaded in a specific multipart upload. You can make this
        request at any time during an in-progress multipart upload
        before you complete the upload (see CompleteMultipartUpload.
        List Parts returns an error for completed uploads. The list
        returned in the List Parts response is sorted by part range.

        The List Parts operation supports pagination. By default, this
        operation returns up to 1,000 uploaded parts in the response.
        You should always check the response for a `marker` at which
        to continue the list; if there are no more items the `marker`
        is `null`. To return a list of parts that begins at a specific
        part, set the `marker` request parameter to the value you
        obtained from a previous List Parts request. You can also
        limit the number of parts returned in the response by
        specifying the `limit` parameter in the request.

        An AWS account has full permission to perform all operations
        (actions). However, AWS Identity and Access Management (IAM)
        users don't have any permissions by default. You must grant
        them explicit permission to perform specific actions. For more
        information, see `Access Control Using AWS Identity and Access
        Management (IAM)`_.

        For conceptual information and the underlying REST API, go to
        `Working with Archives in Amazon Glacier`_ and `List Parts`_
        in the Amazon Glacier Developer Guide .

        :type vault_name: string
        :param vault_name: The name of the vault.

        :type upload_id: string
        :param upload_id: The upload ID of the multipart upload.

        :type marker: string
        :param marker: An opaque string used for pagination. This value
            specifies the part at which the listing of parts should begin. Get
            the marker value from the response of a previous List Parts
            response. You need only include the marker if you are continuing
            the pagination of results started in a previous List Parts request.

        :type limit: string
        :param limit: Specifies the maximum number of parts returned in the
            response body. If this value is not specified, the List Parts
            operation returns up to 1,000 uploads.
        """
    params = {}
    if limit:
        params['limit'] = limit
    if marker:
        params['marker'] = marker
    uri = 'vaults/%s/multipart-uploads/%s' % (vault_name, upload_id)
    return self.make_request('GET', uri, params=params)