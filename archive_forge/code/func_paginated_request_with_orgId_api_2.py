import re
import xml.etree.ElementTree as etree
from io import BytesIO
from copy import deepcopy
from time import sleep
from base64 import b64encode
from typing import Dict
from functools import wraps
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
def paginated_request_with_orgId_api_2(self, action, params=None, data='', headers=None, method='GET', page_size=250):
    """
        A paginated request to the MCP2.0 API
        This essentially calls out to request_with_orgId_api_2 for each page
        and yields the response to make a generator
        This generator can be looped through to grab all the pages.

        :param action: The resource to access (i.e. 'network/vlan')
        :type  action: ``str``

        :param params: Parameters to give to the action
        :type  params: ``dict`` or ``None``

        :param data: The data payload to be added to the request
        :type  data: ``str``

        :param headers: Additional header to be added to the request
        :type  headers: ``str`` or ``dict`` or ``None``

        :param method: HTTP Method for the request (i.e. 'GET', 'POST')
        :type  method: ``str``

        :param page_size: The size of each page to be returned
                          Note: Max page size in MCP2.0 is currently 250
        :type  page_size: ``int``
        """
    if params is None:
        params = {}
    params['pageSize'] = page_size
    resp = self.request_with_orgId_api_2(action, params, data, headers, method).object
    yield resp
    if len(resp) <= 0:
        return
    pcount = resp.get('pageCount')
    psize = resp.get('pageSize')
    pnumber = resp.get('pageNumber')
    while int(pcount) >= int(psize):
        params['pageNumber'] = int(pnumber) + 1
        resp = self.request_with_orgId_api_2(action, params, data, headers, method).object
        pcount = resp.get('pageCount')
        psize = resp.get('pageSize')
        pnumber = resp.get('pageNumber')
        yield resp