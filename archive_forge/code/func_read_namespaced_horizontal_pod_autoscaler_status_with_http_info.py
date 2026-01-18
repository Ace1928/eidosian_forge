from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def read_namespaced_horizontal_pod_autoscaler_status_with_http_info(self, name, namespace, **kwargs):
    """
        read status of the specified HorizontalPodAutoscaler
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread =
        api.read_namespaced_horizontal_pod_autoscaler_status_with_http_info(name,
        namespace, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the HorizontalPodAutoscaler (required)
        :param str namespace: object name and auth scope, such as for teams and
        projects (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :return: V2beta1HorizontalPodAutoscaler
                 If the method is called asynchronously,
                 returns the request thread.
        """
    all_params = ['name', 'namespace', 'pretty']
    all_params.append('async_req')
    all_params.append('_return_http_data_only')
    all_params.append('_preload_content')
    all_params.append('_request_timeout')
    params = locals()
    for key, val in iteritems(params['kwargs']):
        if key not in all_params:
            raise TypeError("Got an unexpected keyword argument '%s' to method read_namespaced_horizontal_pod_autoscaler_status" % key)
        params[key] = val
    del params['kwargs']
    if 'name' not in params or params['name'] is None:
        raise ValueError('Missing the required parameter `name` when calling `read_namespaced_horizontal_pod_autoscaler_status`')
    if 'namespace' not in params or params['namespace'] is None:
        raise ValueError('Missing the required parameter `namespace` when calling `read_namespaced_horizontal_pod_autoscaler_status`')
    collection_formats = {}
    path_params = {}
    if 'name' in params:
        path_params['name'] = params['name']
    if 'namespace' in params:
        path_params['namespace'] = params['namespace']
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
    return self.api_client.call_api('/apis/autoscaling/v2beta1/namespaces/{namespace}/horizontalpodautoscalers/{name}/status', 'GET', path_params, query_params, header_params, body=body_params, post_params=form_params, files=local_var_files, response_type='V2beta1HorizontalPodAutoscaler', auth_settings=auth_settings, async_req=params.get('async_req'), _return_http_data_only=params.get('_return_http_data_only'), _preload_content=params.get('_preload_content', True), _request_timeout=params.get('_request_timeout'), collection_formats=collection_formats)