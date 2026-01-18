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
def parseURL(url, host=None, port=None):
    """
    Return string into URL object.

    URIs are of form 'sip:user@example.com'.
    """
    d = {}
    if not url.startswith('sip:'):
        raise ValueError('unsupported scheme: ' + url[:4])
    parts = url[4:].split(';')
    userdomain, params = (parts[0], parts[1:])
    udparts = userdomain.split('@', 1)
    if len(udparts) == 2:
        userpass, hostport = udparts
        upparts = userpass.split(':', 1)
        if len(upparts) == 1:
            d['username'] = upparts[0]
        else:
            d['username'] = upparts[0]
            d['password'] = upparts[1]
    else:
        hostport = udparts[0]
    hpparts = hostport.split(':', 1)
    if len(hpparts) == 1:
        d['host'] = hpparts[0]
    else:
        d['host'] = hpparts[0]
        d['port'] = int(hpparts[1])
    if host != None:
        d['host'] = host
    if port != None:
        d['port'] = port
    for p in params:
        if p == params[-1] and '?' in p:
            d['headers'] = h = {}
            p, headers = p.split('?', 1)
            for header in headers.split('&'):
                k, v = header.split('=')
                h[k] = v
        nv = p.split('=', 1)
        if len(nv) == 1:
            d.setdefault('other', []).append(p)
            continue
        name, value = nv
        if name == 'user':
            d['usertype'] = value
        elif name in ('transport', 'ttl', 'maddr', 'method', 'tag'):
            if name == 'ttl':
                value = int(value)
            d[name] = value
        else:
            d.setdefault('other', []).append(p)
    return URL(**d)