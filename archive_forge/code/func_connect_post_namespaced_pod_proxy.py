from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def connect_post_namespaced_pod_proxy(self, name, namespace, **kwargs):
    """
        connect POST requests to proxy of Pod
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.connect_post_namespaced_pod_proxy(name, namespace,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the PodProxyOptions (required)
        :param str namespace: object name and auth scope, such as for teams and
        projects (required)
        :param str path: Path is the URL path to use for the current proxy
        request to pod.
        :return: str
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.connect_post_namespaced_pod_proxy_with_http_info(name, namespace, **kwargs)
    else:
        data = self.connect_post_namespaced_pod_proxy_with_http_info(name, namespace, **kwargs)
        return data