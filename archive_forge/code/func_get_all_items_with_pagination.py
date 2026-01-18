from __future__ import (absolute_import, division, print_function)
import json
import os
import time
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.common.parameters import env_fallback
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
def get_all_items_with_pagination(self, uri, query_param=None):
    """
         This implementation mainly to get all available items from ome for pagination
         supported GET uri
        :param uri: uri which supports pagination
        :return: dict.
        """
    try:
        resp = self.invoke_request('GET', uri, query_param=query_param)
        data = resp.json_data
        total_items = data.get('value', [])
        total_count = data.get('@odata.count', 0)
        next_link = data.get('@odata.nextLink', '')
        while next_link:
            resp = self.invoke_request('GET', next_link.split('/api')[-1])
            data = resp.json_data
            value = data['value']
            next_link = data.get('@odata.nextLink', '')
            total_items.extend(value)
        return {'total_count': total_count, 'value': total_items}
    except (URLError, HTTPError, SSLValidationError, ConnectionError, TypeError, ValueError) as err:
        raise err