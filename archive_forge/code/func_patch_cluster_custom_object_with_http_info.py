from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def patch_cluster_custom_object_with_http_info(self, group, version, plural, name, body, **kwargs):
    """
        patch the specified cluster scoped custom object
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.patch_cluster_custom_object_with_http_info(group,
        version, plural, name, body, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str group: the custom resource's group (required)
        :param str version: the custom resource's version (required)
        :param str plural: the custom object's plural name. For TPRs this would
        be lowercase plural kind. (required)
        :param str name: the custom object's name (required)
        :param object body: The JSON schema of the Resource to patch. (required)
        :return: object
                 If the method is called asynchronously,
                 returns the request thread.
        """
    all_params = ['group', 'version', 'plural', 'name', 'body']
    all_params.append('async_req')
    all_params.append('_return_http_data_only')
    all_params.append('_preload_content')
    all_params.append('_request_timeout')
    params = locals()
    for key, val in iteritems(params['kwargs']):
        if key not in all_params:
            raise TypeError("Got an unexpected keyword argument '%s' to method patch_cluster_custom_object" % key)
        params[key] = val
    del params['kwargs']
    if 'group' not in params or params['group'] is None:
        raise ValueError('Missing the required parameter `group` when calling `patch_cluster_custom_object`')
    if 'version' not in params or params['version'] is None:
        raise ValueError('Missing the required parameter `version` when calling `patch_cluster_custom_object`')
    if 'plural' not in params or params['plural'] is None:
        raise ValueError('Missing the required parameter `plural` when calling `patch_cluster_custom_object`')
    if 'name' not in params or params['name'] is None:
        raise ValueError('Missing the required parameter `name` when calling `patch_cluster_custom_object`')
    if 'body' not in params or params['body'] is None:
        raise ValueError('Missing the required parameter `body` when calling `patch_cluster_custom_object`')
    collection_formats = {}
    path_params = {}
    if 'group' in params:
        path_params['group'] = params['group']
    if 'version' in params:
        path_params['version'] = params['version']
    if 'plural' in params:
        path_params['plural'] = params['plural']
    if 'name' in params:
        path_params['name'] = params['name']
    query_params = []
    header_params = {}
    form_params = []
    local_var_files = {}
    body_params = None
    if 'body' in params:
        body_params = params['body']
    header_params['Accept'] = self.api_client.select_header_accept(['application/json'])
    header_params['Content-Type'] = self.api_client.select_header_content_type(['application/merge-patch+json'])
    auth_settings = ['BearerToken']
    return self.api_client.call_api('/apis/{group}/{version}/{plural}/{name}', 'PATCH', path_params, query_params, header_params, body=body_params, post_params=form_params, files=local_var_files, response_type='object', auth_settings=auth_settings, async_req=params.get('async_req'), _return_http_data_only=params.get('_return_http_data_only'), _preload_content=params.get('_preload_content', True), _request_timeout=params.get('_request_timeout'), collection_formats=collection_formats)