import io
import logging
import os
import sys
import urllib
from osc_lib import utils
from openstackclient.api import api
def object_show(self, container=None, object=None):
    """Get object details

        :param string container:
            container name for object to get
        :param string object:
            name of object to get
        :returns:
            dict of object properties
        """
    if container is None or object is None:
        return {}
    response = self._request('HEAD', '%s/%s' % (urllib.parse.quote(container), urllib.parse.quote(object)))
    data = {'account': self._find_account_id(), 'container': container, 'object': object, 'content-type': response.headers.get('content-type')}
    if 'content-length' in response.headers:
        data['content-length'] = response.headers.get('content-length')
    if 'last-modified' in response.headers:
        data['last-modified'] = response.headers.get('last-modified')
    if 'etag' in response.headers:
        data['etag'] = response.headers.get('etag')
    if 'x-object-manifest' in response.headers:
        data['x-object-manifest'] = response.headers.get('x-object-manifest')
    properties = self._get_properties(response.headers, 'x-object-meta-')
    if properties:
        data['properties'] = properties
    return data