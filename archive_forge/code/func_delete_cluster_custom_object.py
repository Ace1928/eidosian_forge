from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def delete_cluster_custom_object(self, group, version, plural, name, body, **kwargs):
    """
        Deletes the specified cluster scoped custom object
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.delete_cluster_custom_object(group, version, plural,
        name, body, async_req=True)
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
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.delete_cluster_custom_object_with_http_info(group, version, plural, name, body, **kwargs)
    else:
        data = self.delete_cluster_custom_object_with_http_info(group, version, plural, name, body, **kwargs)
        return data