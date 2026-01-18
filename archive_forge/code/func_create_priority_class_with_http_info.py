from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def create_priority_class_with_http_info(self, body, **kwargs):
    """
        create a PriorityClass
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.create_priority_class_with_http_info(body,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param V1alpha1PriorityClass body: (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :param str dry_run: When present, indicates that modifications should
        not be persisted. An invalid or unrecognized dryRun directive will
        result in an error response and no further processing of the request.
        Valid values are: - All: all dry run stages will be processed
        :param str field_manager: fieldManager is a name associated with the
        actor or entity that is making these changes. The value must be less
        than or 128 characters long, and only contain printable characters, as
        defined by https://golang.org/pkg/unicode/#IsPrint.
        :return: V1alpha1PriorityClass
                 If the method is called asynchronously,
                 returns the request thread.
        """
    all_params = ['body', 'pretty', 'dry_run', 'field_manager']
    all_params.append('async_req')
    all_params.append('_return_http_data_only')
    all_params.append('_preload_content')
    all_params.append('_request_timeout')
    params = locals()
    for key, val in iteritems(params['kwargs']):
        if key not in all_params:
            raise TypeError("Got an unexpected keyword argument '%s' to method create_priority_class" % key)
        params[key] = val
    del params['kwargs']
    if 'body' not in params or params['body'] is None:
        raise ValueError('Missing the required parameter `body` when calling `create_priority_class`')
    collection_formats = {}
    path_params = {}
    query_params = []
    if 'pretty' in params:
        query_params.append(('pretty', params['pretty']))
    if 'dry_run' in params:
        query_params.append(('dryRun', params['dry_run']))
    if 'field_manager' in params:
        query_params.append(('fieldManager', params['field_manager']))
    header_params = {}
    form_params = []
    local_var_files = {}
    body_params = None
    if 'body' in params:
        body_params = params['body']
    header_params['Accept'] = self.api_client.select_header_accept(['application/json', 'application/yaml', 'application/vnd.kubernetes.protobuf'])
    header_params['Content-Type'] = self.api_client.select_header_content_type(['*/*'])
    auth_settings = ['BearerToken']
    return self.api_client.call_api('/apis/scheduling.k8s.io/v1alpha1/priorityclasses', 'POST', path_params, query_params, header_params, body=body_params, post_params=form_params, files=local_var_files, response_type='V1alpha1PriorityClass', auth_settings=auth_settings, async_req=params.get('async_req'), _return_http_data_only=params.get('_return_http_data_only'), _preload_content=params.get('_preload_content', True), _request_timeout=params.get('_request_timeout'), collection_formats=collection_formats)