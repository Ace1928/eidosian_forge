from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def get_api_resources_with_http_info(self, **kwargs):
    """
        get available resources
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_api_resources_with_http_info(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :return: V1APIResourceList
                 If the method is called asynchronously,
                 returns the request thread.
        """
    all_params = []
    all_params.append('async_req')
    all_params.append('_return_http_data_only')
    all_params.append('_preload_content')
    all_params.append('_request_timeout')
    params = locals()
    for key, val in iteritems(params['kwargs']):
        if key not in all_params:
            raise TypeError("Got an unexpected keyword argument '%s' to method get_api_resources" % key)
        params[key] = val
    del params['kwargs']
    collection_formats = {}
    path_params = {}
    query_params = []
    header_params = {}
    form_params = []
    local_var_files = {}
    body_params = None
    header_params['Accept'] = self.api_client.select_header_accept(['application/json', 'application/yaml', 'application/vnd.kubernetes.protobuf'])
    header_params['Content-Type'] = self.api_client.select_header_content_type(['application/json', 'application/yaml', 'application/vnd.kubernetes.protobuf'])
    auth_settings = ['BearerToken']
    return self.api_client.call_api('/apis/apiregistration.k8s.io/v1/', 'GET', path_params, query_params, header_params, body=body_params, post_params=form_params, files=local_var_files, response_type='V1APIResourceList', auth_settings=auth_settings, async_req=params.get('async_req'), _return_http_data_only=params.get('_return_http_data_only'), _preload_content=params.get('_preload_content', True), _request_timeout=params.get('_request_timeout'), collection_formats=collection_formats)