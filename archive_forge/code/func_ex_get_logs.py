import os
import re
import shlex
import base64
import datetime
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def ex_get_logs(self, container, stream=False):
    """
        Get container logs

        If stream == True, logs will be yielded as a stream
        From Api Version 1.11 and above we need a GET request to get the logs
        Logs are in different format of those of Version 1.10 and below

        :param container: The container to list logs for
        :type  container: :class:`libcloud.container.base.Container`

        :param stream: Stream the output
        :type  stream: ``bool``

        :rtype: ``bool``
        """
    payload = {}
    data = json.dumps(payload)
    if float(self._get_api_version()) > 1.1:
        result = self.connection.request('/v%s/containers/%s/logs?follow=%s&stdout=1&stderr=1' % (self.version, container.id, str(stream))).object
        logs = result
    else:
        result = self.connection.request('/v%s/containers/%s/attach?logs=1&stream=%s&stdout=1&stderr=1' % (self.version, container.id, str(stream)), method='POST', data=data)
        logs = result.body
    return logs