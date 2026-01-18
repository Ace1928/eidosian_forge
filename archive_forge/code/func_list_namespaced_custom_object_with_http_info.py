from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def list_namespaced_custom_object_with_http_info(self, group, version, namespace, plural, **kwargs):
    """
        list or watch namespace scoped custom objects
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.list_namespaced_custom_object_with_http_info(group,
        version, namespace, plural, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str group: The custom resource's group name (required)
        :param str version: The custom resource's version (required)
        :param str namespace: The custom resource's namespace (required)
        :param str plural: The custom resource's plural name. For TPRs this
        would be lowercase plural kind. (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :param str field_selector: A selector to restrict the list of returned
        objects by their fields. Defaults to everything.
        :param str label_selector: A selector to restrict the list of returned
        objects by their labels. Defaults to everything.
        :param str resource_version: When specified with a watch call, shows
        changes that occur after that particular version of a resource. Defaults
        to changes from the beginning of history. When specified for list: - if
        unset, then the result is returned from remote storage based on
        quorum-read flag; - if it's 0, then we simply return what we currently
        have in cache, no guarantee; - if set to non zero, then the result is at
        least as fresh as given rv.
        :param int timeout_seconds: Timeout for the list/watch call. This limits
        the duration of the call, regardless of any activity or inactivity.
        :param bool watch: Watch for changes to the described resources and
        return them as a stream of add, update, and remove notifications.
        :return: object
                 If the method is called asynchronously,
                 returns the request thread.
        """
    all_params = ['group', 'version', 'namespace', 'plural', 'pretty', 'field_selector', 'label_selector', 'resource_version', 'timeout_seconds', 'watch']
    all_params.append('async_req')
    all_params.append('_return_http_data_only')
    all_params.append('_preload_content')
    all_params.append('_request_timeout')
    params = locals()
    for key, val in iteritems(params['kwargs']):
        if key not in all_params:
            raise TypeError("Got an unexpected keyword argument '%s' to method list_namespaced_custom_object" % key)
        params[key] = val
    del params['kwargs']
    if 'group' not in params or params['group'] is None:
        raise ValueError('Missing the required parameter `group` when calling `list_namespaced_custom_object`')
    if 'version' not in params or params['version'] is None:
        raise ValueError('Missing the required parameter `version` when calling `list_namespaced_custom_object`')
    if 'namespace' not in params or params['namespace'] is None:
        raise ValueError('Missing the required parameter `namespace` when calling `list_namespaced_custom_object`')
    if 'plural' not in params or params['plural'] is None:
        raise ValueError('Missing the required parameter `plural` when calling `list_namespaced_custom_object`')
    collection_formats = {}
    path_params = {}
    if 'group' in params:
        path_params['group'] = params['group']
    if 'version' in params:
        path_params['version'] = params['version']
    if 'namespace' in params:
        path_params['namespace'] = params['namespace']
    if 'plural' in params:
        path_params['plural'] = params['plural']
    query_params = []
    if 'pretty' in params:
        query_params.append(('pretty', params['pretty']))
    if 'field_selector' in params:
        query_params.append(('fieldSelector', params['field_selector']))
    if 'label_selector' in params:
        query_params.append(('labelSelector', params['label_selector']))
    if 'resource_version' in params:
        query_params.append(('resourceVersion', params['resource_version']))
    if 'timeout_seconds' in params:
        query_params.append(('timeoutSeconds', params['timeout_seconds']))
    if 'watch' in params:
        query_params.append(('watch', params['watch']))
    header_params = {}
    form_params = []
    local_var_files = {}
    body_params = None
    header_params['Accept'] = self.api_client.select_header_accept(['application/json', 'application/json;stream=watch'])
    header_params['Content-Type'] = self.api_client.select_header_content_type(['*/*'])
    auth_settings = ['BearerToken']
    return self.api_client.call_api('/apis/{group}/{version}/namespaces/{namespace}/{plural}', 'GET', path_params, query_params, header_params, body=body_params, post_params=form_params, files=local_var_files, response_type='object', auth_settings=auth_settings, async_req=params.get('async_req'), _return_http_data_only=params.get('_return_http_data_only'), _preload_content=params.get('_preload_content', True), _request_timeout=params.get('_request_timeout'), collection_formats=collection_formats)