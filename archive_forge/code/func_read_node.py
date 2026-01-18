from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def read_node(self, name, **kwargs):
    """
        read the specified Node
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.read_node(name, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the Node (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :param bool exact: Should the export be exact.  Exact export maintains
        cluster-specific fields like 'Namespace'. Deprecated. Planned for
        removal in 1.18.
        :param bool export: Should this value be exported.  Export strips fields
        that a user can not specify. Deprecated. Planned for removal in 1.18.
        :return: V1Node
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.read_node_with_http_info(name, **kwargs)
    else:
        data = self.read_node_with_http_info(name, **kwargs)
        return data