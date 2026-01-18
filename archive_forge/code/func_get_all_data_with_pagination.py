from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime
import re
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_all_data_with_pagination(ome_obj, uri, query_param=None):
    """To get all the devices with pagination based on the filter provided."""
    query, resp, report_list = ('', None, [])
    try:
        resp = ome_obj.invoke_request('GET', uri, query_param=query_param)
        next_uri = resp.json_data.get('@odata.nextLink', None)
        report_list = resp.json_data.get('value')
        if query_param is not None:
            for k, v in query_param.items():
                query += '{0}={1}'.format(k, v.replace(' ', '%20'))
        while next_uri is not None:
            next_uri_query = '{0}&{1}'.format(next_uri.strip('/api'), query) if query else next_uri.strip('/api')
            resp = ome_obj.invoke_request('GET', next_uri_query)
            report_list.extend(resp.json_data.get('value'))
            next_uri = resp.json_data.get('@odata.nextLink', None)
    except (URLError, HTTPError, SSLValidationError, ConnectionError, TypeError, ValueError) as err:
        raise err
    return {'resp_obj': resp, 'report_list': report_list}