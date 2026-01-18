from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def create_cluster_custom_object(self, group, version, plural, body, **kwargs):
    """
        Creates a cluster scoped Custom object
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.create_cluster_custom_object(group, version, plural,
        body, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str group: The custom resource's group name (required)
        :param str version: The custom resource's version (required)
        :param str plural: The custom resource's plural name. For TPRs this
        would be lowercase plural kind. (required)
        :param object body: The JSON schema of the Resource to create.
        (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :return: object
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.create_cluster_custom_object_with_http_info(group, version, plural, body, **kwargs)
    else:
        data = self.create_cluster_custom_object_with_http_info(group, version, plural, body, **kwargs)
        return data