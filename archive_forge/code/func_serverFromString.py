import os
import re
import socket
import warnings
from typing import Optional, Sequence, Type
from unicodedata import normalize
from zope.interface import directlyProvides, implementer, provider
from constantly import NamedConstant, Names
from incremental import Version
from twisted.internet import defer, error, fdesc, interfaces, threads
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.address import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, ProcessProtocol, Protocol
from twisted.internet._resolver import HostResolution
from twisted.internet.defer import Deferred
from twisted.internet.task import LoopingCall
from twisted.logger import Logger
from twisted.plugin import IPlugin, getPlugins
from twisted.python import deprecate, log
from twisted.python.compat import _matchingString, iterbytes, nativeString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.systemd import ListenFDs
from ._idna import _idnaBytes, _idnaText
def serverFromString(reactor, description):
    """
    Construct a stream server endpoint from an endpoint description string.

    The format for server endpoint descriptions is a simple byte string.  It is
    a prefix naming the type of endpoint, then a colon, then the arguments for
    that endpoint.

    For example, you can call it like this to create an endpoint that will
    listen on TCP port 80::

        serverFromString(reactor, "tcp:80")

    Additional arguments may be specified as keywords, separated with colons.
    For example, you can specify the interface for a TCP server endpoint to
    bind to like this::

        serverFromString(reactor, "tcp:80:interface=127.0.0.1")

    SSL server endpoints may be specified with the 'ssl' prefix, and the
    private key and certificate files may be specified by the C{privateKey} and
    C{certKey} arguments::

        serverFromString(
            reactor, "ssl:443:privateKey=key.pem:certKey=crt.pem")

    If a private key file name (C{privateKey}) isn't provided, a "server.pem"
    file is assumed to exist which contains the private key. If the certificate
    file name (C{certKey}) isn't provided, the private key file is assumed to
    contain the certificate as well.

    You may escape colons in arguments with a backslash, which you will need to
    use if you want to specify a full pathname argument on Windows::

        serverFromString(reactor,
            "ssl:443:privateKey=C\\:/key.pem:certKey=C\\:/cert.pem")

    finally, the 'unix' prefix may be used to specify a filesystem UNIX socket,
    optionally with a 'mode' argument to specify the mode of the socket file
    created by C{listen}::

        serverFromString(reactor, "unix:/var/run/finger")
        serverFromString(reactor, "unix:/var/run/finger:mode=660")

    This function is also extensible; new endpoint types may be registered as
    L{IStreamServerEndpointStringParser} plugins.  See that interface for more
    information.

    @param reactor: The server endpoint will be constructed with this reactor.

    @param description: The strports description to parse.
    @type description: L{str}

    @return: A new endpoint which can be used to listen with the parameters
        given by C{description}.

    @rtype: L{IStreamServerEndpoint<twisted.internet.interfaces.IStreamServerEndpoint>}

    @raise ValueError: when the 'description' string cannot be parsed.

    @since: 10.2
    """
    nameOrPlugin, args, kw = _parseServer(description, None)
    if type(nameOrPlugin) is not str:
        plugin = nameOrPlugin
        return plugin.parseStreamServer(reactor, *args, **kw)
    else:
        name = nameOrPlugin
    args = args[:1] + args[2:]
    return _endpointServerFactories[name](reactor, *args, **kw)