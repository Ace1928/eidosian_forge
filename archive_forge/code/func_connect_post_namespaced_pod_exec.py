from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def connect_post_namespaced_pod_exec(self, name, namespace, **kwargs):
    """
        connect POST requests to exec of Pod
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.connect_post_namespaced_pod_exec(name, namespace,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the PodExecOptions (required)
        :param str namespace: object name and auth scope, such as for teams and
        projects (required)
        :param str command: Command is the remote command to execute. argv
        array. Not executed within a shell.
        :param str container: Container in which to execute the command.
        Defaults to only container if there is only one container in the pod.
        :param bool stderr: Redirect the standard error stream of the pod for
        this call. Defaults to true.
        :param bool stdin: Redirect the standard input stream of the pod for
        this call. Defaults to false.
        :param bool stdout: Redirect the standard output stream of the pod for
        this call. Defaults to true.
        :param bool tty: TTY if true indicates that a tty will be allocated for
        the exec call. Defaults to false.
        :return: str
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.connect_post_namespaced_pod_exec_with_http_info(name, namespace, **kwargs)
    else:
        data = self.connect_post_namespaced_pod_exec_with_http_info(name, namespace, **kwargs)
        return data