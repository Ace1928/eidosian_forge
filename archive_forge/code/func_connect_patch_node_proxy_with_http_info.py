from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def connect_patch_node_proxy_with_http_info(self, name, **kwargs):
    """
        connect PATCH requests to proxy of Node
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.connect_patch_node_proxy_with_http_info(name,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the NodeProxyOptions (required)
        :param str path: Path is the URL path to use for the current proxy
        request to node.
        :return: str
                 If the method is called asynchronously,
                 returns the request thread.
        """
    all_params = ['name', 'path']
    all_params.append('async_req')
    all_params.append('_return_http_data_only')
    all_params.append('_preload_content')
    all_params.append('_request_timeout')
    params = locals()
    for key, val in iteritems(params['kwargs']):
        if key not in all_params:
            raise TypeError("Got an unexpected keyword argument '%s' to method connect_patch_node_proxy" % key)
        params[key] = val
    del params['kwargs']
    if 'name' not in params or params['name'] is None:
        raise ValueError('Missing the required parameter `name` when calling `connect_patch_node_proxy`')
    collection_formats = {}
    path_params = {}
    if 'name' in params:
        path_params['name'] = params['name']
    query_params = []
    if 'path' in params:
        query_params.append(('path', params['path']))
    header_params = {}
    form_params = []
    local_var_files = {}
    body_params = None
    header_params['Accept'] = self.api_client.select_header_accept(['*/*'])
    header_params['Content-Type'] = self.api_client.select_header_content_type(['*/*'])
    auth_settings = ['BearerToken']
    return self.api_client.call_api('/api/v1/nodes/{name}/proxy', 'PATCH', path_params, query_params, header_params, body=body_params, post_params=form_params, files=local_var_files, response_type='str', auth_settings=auth_settings, async_req=params.get('async_req'), _return_http_data_only=params.get('_return_http_data_only'), _preload_content=params.get('_preload_content', True), _request_timeout=params.get('_request_timeout'), collection_formats=collection_formats)