from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def read_custom_resource_definition_status_with_http_info(self, name, **kwargs):
    """
        read status of the specified CustomResourceDefinition
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread =
        api.read_custom_resource_definition_status_with_http_info(name,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the CustomResourceDefinition (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :return: V1beta1CustomResourceDefinition
                 If the method is called asynchronously,
                 returns the request thread.
        """
    all_params = ['name', 'pretty']
    all_params.append('async_req')
    all_params.append('_return_http_data_only')
    all_params.append('_preload_content')
    all_params.append('_request_timeout')
    params = locals()
    for key, val in iteritems(params['kwargs']):
        if key not in all_params:
            raise TypeError("Got an unexpected keyword argument '%s' to method read_custom_resource_definition_status" % key)
        params[key] = val
    del params['kwargs']
    if 'name' not in params or params['name'] is None:
        raise ValueError('Missing the required parameter `name` when calling `read_custom_resource_definition_status`')
    collection_formats = {}
    path_params = {}
    if 'name' in params:
        path_params['name'] = params['name']
    query_params = []
    if 'pretty' in params:
        query_params.append(('pretty', params['pretty']))
    header_params = {}
    form_params = []
    local_var_files = {}
    body_params = None
    header_params['Accept'] = self.api_client.select_header_accept(['application/json', 'application/yaml', 'application/vnd.kubernetes.protobuf'])
    header_params['Content-Type'] = self.api_client.select_header_content_type(['*/*'])
    auth_settings = ['BearerToken']
    return self.api_client.call_api('/apis/apiextensions.k8s.io/v1beta1/customresourcedefinitions/{name}/status', 'GET', path_params, query_params, header_params, body=body_params, post_params=form_params, files=local_var_files, response_type='V1beta1CustomResourceDefinition', auth_settings=auth_settings, async_req=params.get('async_req'), _return_http_data_only=params.get('_return_http_data_only'), _preload_content=params.get('_preload_content', True), _request_timeout=params.get('_request_timeout'), collection_formats=collection_formats)