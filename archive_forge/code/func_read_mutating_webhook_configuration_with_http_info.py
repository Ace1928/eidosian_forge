from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def read_mutating_webhook_configuration_with_http_info(self, name, **kwargs):
    """
        read the specified MutatingWebhookConfiguration
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread =
        api.read_mutating_webhook_configuration_with_http_info(name,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the MutatingWebhookConfiguration (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :param bool exact: Should the export be exact.  Exact export maintains
        cluster-specific fields like 'Namespace'. Deprecated. Planned for
        removal in 1.18.
        :param bool export: Should this value be exported.  Export strips fields
        that a user can not specify. Deprecated. Planned for removal in 1.18.
        :return: V1beta1MutatingWebhookConfiguration
                 If the method is called asynchronously,
                 returns the request thread.
        """
    all_params = ['name', 'pretty', 'exact', 'export']
    all_params.append('async_req')
    all_params.append('_return_http_data_only')
    all_params.append('_preload_content')
    all_params.append('_request_timeout')
    params = locals()
    for key, val in iteritems(params['kwargs']):
        if key not in all_params:
            raise TypeError("Got an unexpected keyword argument '%s' to method read_mutating_webhook_configuration" % key)
        params[key] = val
    del params['kwargs']
    if 'name' not in params or params['name'] is None:
        raise ValueError('Missing the required parameter `name` when calling `read_mutating_webhook_configuration`')
    collection_formats = {}
    path_params = {}
    if 'name' in params:
        path_params['name'] = params['name']
    query_params = []
    if 'pretty' in params:
        query_params.append(('pretty', params['pretty']))
    if 'exact' in params:
        query_params.append(('exact', params['exact']))
    if 'export' in params:
        query_params.append(('export', params['export']))
    header_params = {}
    form_params = []
    local_var_files = {}
    body_params = None
    header_params['Accept'] = self.api_client.select_header_accept(['application/json', 'application/yaml', 'application/vnd.kubernetes.protobuf'])
    header_params['Content-Type'] = self.api_client.select_header_content_type(['*/*'])
    auth_settings = ['BearerToken']
    return self.api_client.call_api('/apis/admissionregistration.k8s.io/v1beta1/mutatingwebhookconfigurations/{name}', 'GET', path_params, query_params, header_params, body=body_params, post_params=form_params, files=local_var_files, response_type='V1beta1MutatingWebhookConfiguration', auth_settings=auth_settings, async_req=params.get('async_req'), _return_http_data_only=params.get('_return_http_data_only'), _preload_content=params.get('_preload_content', True), _request_timeout=params.get('_request_timeout'), collection_formats=collection_formats)