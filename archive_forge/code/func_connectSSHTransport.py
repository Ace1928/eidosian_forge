from __future__ import annotations
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.trial.unittest import TestCase
def connectSSHTransport(service: SSHService, hostAddress: interfaces.IAddress | None=None, peerAddress: interfaces.IAddress | None=None) -> None:
    """
    Connect a SSHTransport which is already connected to a remote peer to
    the channel under test.

    @param service: Service used over the connected transport.
    @type service: L{SSHService}

    @param hostAddress: Local address of the connected transport.
    @type hostAddress: L{interfaces.IAddress}

    @param peerAddress: Remote address of the connected transport.
    @type peerAddress: L{interfaces.IAddress}
    """
    transport = SSHServerTransport()
    transport.makeConnection(StringTransport(hostAddress=hostAddress, peerAddress=peerAddress))
    transport.setService(service)