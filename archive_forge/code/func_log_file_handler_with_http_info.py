from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def log_file_handler_with_http_info(self, logpath, **kwargs):
    """
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.log_file_handler_with_http_info(logpath,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str logpath: path to the log (required)
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
    all_params = ['logpath']
    all_params.append('async_req')
    all_params.append('_return_http_data_only')
    all_params.append('_preload_content')
    all_params.append('_request_timeout')
    params = locals()
    for key, val in iteritems(params['kwargs']):
        if key not in all_params:
            raise TypeError("Got an unexpected keyword argument '%s' to method log_file_handler" % key)
        params[key] = val
    del params['kwargs']
    if 'logpath' not in params or params['logpath'] is None:
        raise ValueError('Missing the required parameter `logpath` when calling `log_file_handler`')
    collection_formats = {}
    path_params = {}
    if 'logpath' in params:
        path_params['logpath'] = params['logpath']
    query_params = []
    header_params = {}
    form_params = []
    local_var_files = {}
    body_params = None
    auth_settings = ['BearerToken']
    return self.api_client.call_api('/logs/{logpath}', 'GET', path_params, query_params, header_params, body=body_params, post_params=form_params, files=local_var_files, response_type=None, auth_settings=auth_settings, async_req=params.get('async_req'), _return_http_data_only=params.get('_return_http_data_only'), _preload_content=params.get('_preload_content', True), _request_timeout=params.get('_request_timeout'), collection_formats=collection_formats)