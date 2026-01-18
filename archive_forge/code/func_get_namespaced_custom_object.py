from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def get_namespaced_custom_object(self, group, version, namespace, plural, name, **kwargs):
    """
        Returns a namespace scoped custom object
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_namespaced_custom_object(group, version, namespace,
        plural, name, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str group: the custom resource's group (required)
        :param str version: the custom resource's version (required)
        :param str namespace: The custom resource's namespace (required)
        :param str plural: the custom resource's plural name. For TPRs this
        would be lowercase plural kind. (required)
        :param str name: the custom object's name (required)
        :return: object
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.get_namespaced_custom_object_with_http_info(group, version, namespace, plural, name, **kwargs)
    else:
        data = self.get_namespaced_custom_object_with_http_info(group, version, namespace, plural, name, **kwargs)
        return data