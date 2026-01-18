import struct
from twisted.conch.ssh import channel, common
from twisted.internet import protocol, reactor
from twisted.internet.endpoints import HostnameEndpoint, connectProtocol
def packOpen_direct_tcpip(destination, source):
    """
    Pack the data suitable for sending in a CHANNEL_OPEN packet.

    @type destination: L{tuple}
    @param destination: A tuple of the (host, port) of the destination host.

    @type source: L{tuple}
    @param source: A tuple of the (host, port) of the source host.
    """
    connHost, connPort = destination
    origHost, origPort = source
    if isinstance(connHost, str):
        connHost = connHost.encode('utf-8')
    if isinstance(origHost, str):
        origHost = origHost.encode('utf-8')
    conn = common.NS(connHost) + struct.pack('>L', connPort)
    orig = common.NS(origHost) + struct.pack('>L', origPort)
    return conn + orig