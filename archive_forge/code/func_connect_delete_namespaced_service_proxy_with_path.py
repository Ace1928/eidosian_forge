from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def connect_delete_namespaced_service_proxy_with_path(self, name, namespace, path, **kwargs):
    """
        connect DELETE requests to proxy of Service
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.connect_delete_namespaced_service_proxy_with_path(name,
        namespace, path, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the ServiceProxyOptions (required)
        :param str namespace: object name and auth scope, such as for teams and
        projects (required)
        :param str path: path to the resource (required)
        :param str path2: Path is the part of URLs that include service
        endpoints, suffixes, and parameters to use for the current proxy request
        to service. For example, the whole request URL is
        http://localhost/api/v1/namespaces/kube-system/services/elasticsearch-logging/_search?q=user:kimchy.
        Path is _search?q=user:kimchy.
        :return: str
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.connect_delete_namespaced_service_proxy_with_path_with_http_info(name, namespace, path, **kwargs)
    else:
        data = self.connect_delete_namespaced_service_proxy_with_path_with_http_info(name, namespace, path, **kwargs)
        return data