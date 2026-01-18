from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def delete_cluster_custom_object_with_http_info(self, group, version, plural, name, body, **kwargs):
    """
        Deletes the specified cluster scoped custom object
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.delete_cluster_custom_object_with_http_info(group,
        version, plural, name, body, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str group: the custom resource's group (required)
        :param str version: the custom resource's version (required)
        :param str plural: the custom object's plural name. For TPRs this would
        be lowercase plural kind. (required)
        :param str name: the custom object's name (required)
        :param V1DeleteOptions body: (required)
        :param int grace_period_seconds: The duration in seconds before the
        object should be deleted. Value must be non-negative integer. The value
        zero indicates delete immediately. If this value is nil, the default
        grace period for the specified type will be used. Defaults to a per
        object value if not specified. zero means delete immediately.
        :param bool orphan_dependents: Deprecated: please use the
        PropagationPolicy, this field will be deprecated in 1.7. Should the
        dependent objects be orphaned. If true/false, the "orphan" finalizer
        will be added to/removed from the object's finalizers list. Either this
        field or PropagationPolicy may be set, but not both.
        :param str propagation_policy: Whether and how garbage collection will
        be performed. Either this field or OrphanDependents may be set, but not
        both. The default policy is decided by the existing finalizer set in the
        metadata.finalizers and the resource-specific default policy.
        :return: object
                 If the method is called asynchronously,
                 returns the request thread.
        """
    all_params = ['group', 'version', 'plural', 'name', 'body', 'grace_period_seconds', 'orphan_dependents', 'propagation_policy']
    all_params.append('async_req')
    all_params.append('_return_http_data_only')
    all_params.append('_preload_content')
    all_params.append('_request_timeout')
    params = locals()
    for key, val in iteritems(params['kwargs']):
        if key not in all_params:
            raise TypeError("Got an unexpected keyword argument '%s' to method delete_cluster_custom_object" % key)
        params[key] = val
    del params['kwargs']
    if 'group' not in params or params['group'] is None:
        raise ValueError('Missing the required parameter `group` when calling `delete_cluster_custom_object`')
    if 'version' not in params or params['version'] is None:
        raise ValueError('Missing the required parameter `version` when calling `delete_cluster_custom_object`')
    if 'plural' not in params or params['plural'] is None:
        raise ValueError('Missing the required parameter `plural` when calling `delete_cluster_custom_object`')
    if 'name' not in params or params['name'] is None:
        raise ValueError('Missing the required parameter `name` when calling `delete_cluster_custom_object`')
    if 'body' not in params or params['body'] is None:
        raise ValueError('Missing the required parameter `body` when calling `delete_cluster_custom_object`')
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
    if 'grace_period_seconds' in params:
        query_params.append(('gracePeriodSeconds', params['grace_period_seconds']))
    if 'orphan_dependents' in params:
        query_params.append(('orphanDependents', params['orphan_dependents']))
    if 'propagation_policy' in params:
        query_params.append(('propagationPolicy', params['propagation_policy']))
    header_params = {}
    form_params = []
    local_var_files = {}
    body_params = None
    if 'body' in params:
        body_params = params['body']
    header_params['Accept'] = self.api_client.select_header_accept(['application/json'])
    header_params['Content-Type'] = self.api_client.select_header_content_type(['*/*'])
    auth_settings = ['BearerToken']
    return self.api_client.call_api('/apis/{group}/{version}/{plural}/{name}', 'DELETE', path_params, query_params, header_params, body=body_params, post_params=form_params, files=local_var_files, response_type='object', auth_settings=auth_settings, async_req=params.get('async_req'), _return_http_data_only=params.get('_return_http_data_only'), _preload_content=params.get('_preload_content', True), _request_timeout=params.get('_request_timeout'), collection_formats=collection_formats)