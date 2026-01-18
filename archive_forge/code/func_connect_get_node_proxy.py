from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def connect_get_node_proxy(self, name, **kwargs):
    """
        connect GET requests to proxy of Node
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.connect_get_node_proxy(name, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the NodeProxyOptions (required)
        :param str path: Path is the URL path to use for the current proxy
        request to node.
        :return: str
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.connect_get_node_proxy_with_http_info(name, **kwargs)
    else:
        data = self.connect_get_node_proxy_with_http_info(name, **kwargs)
        return data