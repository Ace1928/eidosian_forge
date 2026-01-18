import base64
import functools
import operator
import time
import iso8601
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack.compute.v2._proxy import Proxy
from openstack.compute.v2 import quota_set as _qs
from openstack.compute.v2 import server as _server
from openstack import exceptions
from openstack import utils
def get_server_console(self, server, length=None):
    """Get the console log for a server.

        :param server: The server to fetch the console log for. Can be either
            a server dict or the Name or ID of the server.
        :param int length: The number of lines you would like to retrieve from
            the end of the log. (optional, defaults to all)

        :returns: A string containing the text of the console log or an
            empty string if the cloud does not support console logs.
        :raises: :class:`~openstack.exceptions.SDKException` if an invalid
            server argument is given or if something else unforseen happens
        """
    if not isinstance(server, dict):
        server = self.get_server(server, bare=True)
    if not server:
        raise exceptions.SDKException('Console log requested for invalid server')
    try:
        return self._get_server_console_output(server['id'], length)
    except exceptions.BadRequestException:
        return ''