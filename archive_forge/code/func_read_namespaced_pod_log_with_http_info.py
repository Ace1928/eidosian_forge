from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def read_namespaced_pod_log_with_http_info(self, name, namespace, **kwargs):
    """
        read log of the specified Pod
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.read_namespaced_pod_log_with_http_info(name, namespace,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the Pod (required)
        :param str namespace: object name and auth scope, such as for teams and
        projects (required)
        :param str container: The container for which to stream logs. Defaults
        to only container if there is one container in the pod.
        :param bool follow: Follow the log stream of the pod. Defaults to false.
        :param int limit_bytes: If set, the number of bytes to read from the
        server before terminating the log output. This may not display a
        complete final line of logging, and may return slightly more or slightly
        less than the specified limit.
        :param str pretty: If 'true', then the output is pretty printed.
        :param bool previous: Return previous terminated container logs.
        Defaults to false.
        :param int since_seconds: A relative time in seconds before the current
        time from which to show logs. If this value precedes the time a pod was
        started, only logs since the pod start will be returned. If this value
        is in the future, no logs will be returned. Only one of sinceSeconds or
        sinceTime may be specified.
        :param int tail_lines: If set, the number of lines from the end of the
        logs to show. If not specified, logs are shown from the creation of the
        container or sinceSeconds or sinceTime
        :param bool timestamps: If true, add an RFC3339 or RFC3339Nano timestamp
        at the beginning of every line of log output. Defaults to false.
        :return: str
                 If the method is called asynchronously,
                 returns the request thread.
        """
    all_params = ['name', 'namespace', 'container', 'follow', 'limit_bytes', 'pretty', 'previous', 'since_seconds', 'tail_lines', 'timestamps']
    all_params.append('async_req')
    all_params.append('_return_http_data_only')
    all_params.append('_preload_content')
    all_params.append('_request_timeout')
    params = locals()
    for key, val in iteritems(params['kwargs']):
        if key not in all_params:
            raise TypeError("Got an unexpected keyword argument '%s' to method read_namespaced_pod_log" % key)
        params[key] = val
    del params['kwargs']
    if 'name' not in params or params['name'] is None:
        raise ValueError('Missing the required parameter `name` when calling `read_namespaced_pod_log`')
    if 'namespace' not in params or params['namespace'] is None:
        raise ValueError('Missing the required parameter `namespace` when calling `read_namespaced_pod_log`')
    collection_formats = {}
    path_params = {}
    if 'name' in params:
        path_params['name'] = params['name']
    if 'namespace' in params:
        path_params['namespace'] = params['namespace']
    query_params = []
    if 'container' in params:
        query_params.append(('container', params['container']))
    if 'follow' in params:
        query_params.append(('follow', params['follow']))
    if 'limit_bytes' in params:
        query_params.append(('limitBytes', params['limit_bytes']))
    if 'pretty' in params:
        query_params.append(('pretty', params['pretty']))
    if 'previous' in params:
        query_params.append(('previous', params['previous']))
    if 'since_seconds' in params:
        query_params.append(('sinceSeconds', params['since_seconds']))
    if 'tail_lines' in params:
        query_params.append(('tailLines', params['tail_lines']))
    if 'timestamps' in params:
        query_params.append(('timestamps', params['timestamps']))
    header_params = {}
    form_params = []
    local_var_files = {}
    body_params = None
    header_params['Accept'] = self.api_client.select_header_accept(['text/plain', 'application/json', 'application/yaml', 'application/vnd.kubernetes.protobuf'])
    header_params['Content-Type'] = self.api_client.select_header_content_type(['*/*'])
    auth_settings = ['BearerToken']
    return self.api_client.call_api('/api/v1/namespaces/{namespace}/pods/{name}/log', 'GET', path_params, query_params, header_params, body=body_params, post_params=form_params, files=local_var_files, response_type='str', auth_settings=auth_settings, async_req=params.get('async_req'), _return_http_data_only=params.get('_return_http_data_only'), _preload_content=params.get('_preload_content', True), _request_timeout=params.get('_request_timeout'), collection_formats=collection_formats)