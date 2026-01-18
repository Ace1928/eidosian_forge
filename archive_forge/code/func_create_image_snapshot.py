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
def create_image_snapshot(self, name, server, wait=False, timeout=3600, **metadata):
    """Create an image by snapshotting an existing server.

        ..note::
            On most clouds this is a cold snapshot - meaning that the server in
            question will be shutdown before taking the snapshot. It is
            possible that it's a live snapshot - but there is no way to know as
            a user, so caveat emptor.

        :param name: Name of the image to be created
        :param server: Server name or ID or dict representing the server
            to be snapshotted
        :param wait: If true, waits for image to be created.
        :param timeout: Seconds to wait for image creation. None is forever.
        :param metadata: Metadata to give newly-created image entity

        :returns: The created image ``Image`` object.
        :raises: :class:`~openstack.exceptions.SDKException` if there are
            problems uploading
        """
    if not isinstance(server, dict):
        server_obj = self.get_server(server, bare=True)
        if not server_obj:
            raise exceptions.SDKException('Server {server} could not be found and therefore could not be snapshotted.'.format(server=server))
        server = server_obj
    image = self.compute.create_server_image(server, name=name, metadata=metadata, wait=wait, timeout=timeout)
    return image