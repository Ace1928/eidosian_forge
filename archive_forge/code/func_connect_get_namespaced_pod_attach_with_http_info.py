from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def connect_get_namespaced_pod_attach_with_http_info(self, name, namespace, **kwargs):
    """
        connect GET requests to attach of Pod
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.connect_get_namespaced_pod_attach_with_http_info(name,
        namespace, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the PodAttachOptions (required)
        :param str namespace: object name and auth scope, such as for teams and
        projects (required)
        :param str container: The container in which to execute the command.
        Defaults to only container if there is only one container in the pod.
        :param bool stderr: Stderr if true indicates that stderr is to be
        redirected for the attach call. Defaults to true.
        :param bool stdin: Stdin if true, redirects the standard input stream of
        the pod for this call. Defaults to false.
        :param bool stdout: Stdout if true indicates that stdout is to be
        redirected for the attach call. Defaults to true.
        :param bool tty: TTY if true indicates that a tty will be allocated for
        the attach call. This is passed through the container runtime so the tty
        is allocated on the worker node by the container runtime. Defaults to
        false.
        :return: str
                 If the method is called asynchronously,
                 returns the request thread.
        """
    all_params = ['name', 'namespace', 'container', 'stderr', 'stdin', 'stdout', 'tty']
    all_params.append('async_req')
    all_params.append('_return_http_data_only')
    all_params.append('_preload_content')
    all_params.append('_request_timeout')
    params = locals()
    for key, val in iteritems(params['kwargs']):
        if key not in all_params:
            raise TypeError("Got an unexpected keyword argument '%s' to method connect_get_namespaced_pod_attach" % key)
        params[key] = val
    del params['kwargs']
    if 'name' not in params or params['name'] is None:
        raise ValueError('Missing the required parameter `name` when calling `connect_get_namespaced_pod_attach`')
    if 'namespace' not in params or params['namespace'] is None:
        raise ValueError('Missing the required parameter `namespace` when calling `connect_get_namespaced_pod_attach`')
    collection_formats = {}
    path_params = {}
    if 'name' in params:
        path_params['name'] = params['name']
    if 'namespace' in params:
        path_params['namespace'] = params['namespace']
    query_params = []
    if 'container' in params:
        query_params.append(('container', params['container']))
    if 'stderr' in params:
        query_params.append(('stderr', params['stderr']))
    if 'stdin' in params:
        query_params.append(('stdin', params['stdin']))
    if 'stdout' in params:
        query_params.append(('stdout', params['stdout']))
    if 'tty' in params:
        query_params.append(('tty', params['tty']))
    header_params = {}
    form_params = []
    local_var_files = {}
    body_params = None
    header_params['Accept'] = self.api_client.select_header_accept(['*/*'])
    header_params['Content-Type'] = self.api_client.select_header_content_type(['*/*'])
    auth_settings = ['BearerToken']
    return self.api_client.call_api('/api/v1/namespaces/{namespace}/pods/{name}/attach', 'GET', path_params, query_params, header_params, body=body_params, post_params=form_params, files=local_var_files, response_type='str', auth_settings=auth_settings, async_req=params.get('async_req'), _return_http_data_only=params.get('_return_http_data_only'), _preload_content=params.get('_preload_content', True), _request_timeout=params.get('_request_timeout'), collection_formats=collection_formats)