import contextlib
import getpass
import io
import os
import sys
from base64 import decodebytes
from twisted.conch.client import agent
from twisted.conch.client.knownhosts import ConsoleUI, KnownHostsFile
from twisted.conch.error import ConchError
from twisted.conch.ssh import common, keys, userauth
from twisted.internet import defer, protocol, reactor
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
def getHostKeyAlgorithms(host, options):
    """
    Look in known_hosts for a key corresponding to C{host}.
    This can be used to change the order of supported key types
    in the KEXINIT packet.

    @type host: L{str}
    @param host: the host to check in known_hosts
    @type options: L{twisted.conch.client.options.ConchOptions}
    @param options: options passed to client
    @return: L{list} of L{str} representing key types or L{None}.
    """
    knownHosts = KnownHostsFile.fromPath(FilePath(options['known-hosts'] or os.path.expanduser(_KNOWN_HOSTS)))
    keyTypes = []
    for entry in knownHosts.iterentries():
        if entry.matchesHost(host):
            if entry.keyType not in keyTypes:
                keyTypes.append(entry.keyType)
    return keyTypes or None