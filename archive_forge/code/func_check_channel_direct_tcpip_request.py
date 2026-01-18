import threading
from paramiko import util
from paramiko.common import (
def check_channel_direct_tcpip_request(self, chanid, origin, destination):
    """
        Determine if a local port forwarding channel will be granted, and
        return ``OPEN_SUCCEEDED`` or an error code.  This method is
        called in server mode when the client requests a channel, after
        authentication is complete.

        The ``chanid`` parameter is a small number that uniquely identifies the
        channel within a `.Transport`.  A `.Channel` object is not created
        unless this method returns ``OPEN_SUCCEEDED`` -- once a
        `.Channel` object is created, you can call `.Channel.get_id` to
        retrieve the channel ID.

        The origin and destination parameters are (ip_address, port) tuples
        that correspond to both ends of the TCP connection in the forwarding
        tunnel.

        The return value should either be ``OPEN_SUCCEEDED`` (or
        ``0``) to allow the channel request, or one of the following error
        codes to reject it:

            - ``OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED``
            - ``OPEN_FAILED_CONNECT_FAILED``
            - ``OPEN_FAILED_UNKNOWN_CHANNEL_TYPE``
            - ``OPEN_FAILED_RESOURCE_SHORTAGE``

        The default implementation always returns
        ``OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED``.

        :param int chanid: ID of the channel
        :param tuple origin:
            2-tuple containing the IP address and port of the originator
            (client side)
        :param tuple destination:
            2-tuple containing the IP address and port of the destination
            (server side)
        :return: an `int` success or failure code (listed above)
        """
    return OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED