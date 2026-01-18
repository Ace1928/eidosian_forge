from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def connect_patch_namespaced_pod_proxy_with_path(self, name, namespace, path, **kwargs):
    """
        connect PATCH requests to proxy of Pod
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.connect_patch_namespaced_pod_proxy_with_path(name,
        namespace, path, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the PodProxyOptions (required)
        :param str namespace: object name and auth scope, such as for teams and
        projects (required)
        :param str path: path to the resource (required)
        :param str path2: Path is the URL path to use for the current proxy
        request to pod.
        :return: str
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.connect_patch_namespaced_pod_proxy_with_path_with_http_info(name, namespace, path, **kwargs)
    else:
        data = self.connect_patch_namespaced_pod_proxy_with_path_with_http_info(name, namespace, path, **kwargs)
        return data