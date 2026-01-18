import socket
import time
import warnings
from collections import OrderedDict
from typing import Dict, List
from zope.interface import Interface, implementer
from twisted import cred
from twisted.internet import defer, protocol, reactor
from twisted.protocols import basic
from twisted.python import log
def parseAddress(address, host=None, port=None, clean=0):
    """
    Return (name, uri, params) for From/To/Contact header.

    @param clean: remove unnecessary info, usually for From and To headers.
    """
    address = address.strip()
    if address.startswith('sip:'):
        return ('', parseURL(address, host=host, port=port), {})
    params = {}
    name, url = address.split('<', 1)
    name = name.strip()
    if name.startswith('"'):
        name = name[1:]
    if name.endswith('"'):
        name = name[:-1]
    url, paramstring = url.split('>', 1)
    url = parseURL(url, host=host, port=port)
    paramstring = paramstring.strip()
    if paramstring:
        for l in paramstring.split(';'):
            if not l:
                continue
            k, v = l.split('=')
            params[k] = v
    if clean:
        url.ttl = None
        url.headers = {}
        url.transport = None
        url.maddr = None
    return (name, url, params)