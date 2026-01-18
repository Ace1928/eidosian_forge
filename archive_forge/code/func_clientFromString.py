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
def clientFromString(reactor, description):
    """
    Construct a client endpoint from a description string.

    Client description strings are much like server description strings,
    although they take all of their arguments as keywords, aside from host and
    port.

    You can create a TCP client endpoint with the 'host' and 'port' arguments,
    like so::

        clientFromString(reactor, "tcp:host=www.example.com:port=80")

    or, without specifying host and port keywords::

        clientFromString(reactor, "tcp:www.example.com:80")

    Or you can specify only one or the other, as in the following 2 examples::

        clientFromString(reactor, "tcp:host=www.example.com:80")
        clientFromString(reactor, "tcp:www.example.com:port=80")

    or an SSL client endpoint with those arguments, plus the arguments used by
    the server SSL, for a client certificate::

        clientFromString(reactor, "ssl:web.example.com:443:"
                                  "privateKey=foo.pem:certKey=foo.pem")

    to specify your certificate trust roots, you can identify a directory with
    PEM files in it with the C{caCertsDir} argument::

        clientFromString(reactor, "ssl:host=web.example.com:port=443:"
                                  "caCertsDir=/etc/ssl/certs")

    Both TCP and SSL client endpoint description strings can include a
    'bindAddress' keyword argument, whose value should be a local IPv4
    address. This fixes the client socket to that IP address::

        clientFromString(reactor, "tcp:www.example.com:80:"
                                  "bindAddress=192.0.2.100")

    NB: Fixed client ports are not currently supported in TCP or SSL
    client endpoints. The client socket will always use an ephemeral
    port assigned by the operating system

    You can create a UNIX client endpoint with the 'path' argument and optional
    'lockfile' and 'timeout' arguments::

        clientFromString(
            reactor, b"unix:path=/var/foo/bar:lockfile=1:timeout=9")

    or, with the path as a positional argument with or without optional
    arguments as in the following 2 examples::

        clientFromString(reactor, "unix:/var/foo/bar")
        clientFromString(reactor, "unix:/var/foo/bar:lockfile=1:timeout=9")

    This function is also extensible; new endpoint types may be registered as
    L{IStreamClientEndpointStringParserWithReactor} plugins.  See that
    interface for more information.

    @param reactor: The client endpoint will be constructed with this reactor.

    @param description: The strports description to parse.
    @type description: L{str}

    @return: A new endpoint which can be used to connect with the parameters
        given by C{description}.
    @rtype: L{IStreamClientEndpoint<twisted.internet.interfaces.IStreamClientEndpoint>}

    @since: 10.2
    """
    args, kwargs = _parse(description)
    aname = args.pop(0)
    name = aname.upper()
    if name not in _clientParsers:
        plugin = _matchPluginToPrefix(getPlugins(IStreamClientEndpointStringParserWithReactor), name)
        return plugin.parseStreamClient(reactor, *args, **kwargs)
    kwargs = _clientParsers[name](*args, **kwargs)
    return _endpointClientFactories[name](reactor, **kwargs)