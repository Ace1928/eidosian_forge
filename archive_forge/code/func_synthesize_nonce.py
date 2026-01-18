import time
import functools
from hashlib import md5
from urllib.request import parse_http_list, parse_keqv_list
import cherrypy
from cherrypy._cpcompat import ntob, tonative
def synthesize_nonce(s, key, timestamp=None):
    """Synthesize a nonce value which resists spoofing and can be checked
    for staleness. Returns a string suitable as the value for 'nonce' in
    the www-authenticate header.

    s
        A string related to the resource, such as the hostname of the server.

    key
        A secret string known only to the server.

    timestamp
        An integer seconds-since-the-epoch timestamp

    """
    if timestamp is None:
        timestamp = int(time.time())
    h = md5_hex('%s:%s:%s' % (timestamp, s, key))
    nonce = '%s:%s' % (timestamp, h)
    return nonce