import os
import re
import time
import hashlib
import threading
import warnings
from base64 import b64encode
from .compat import urlparse, str, basestring
from .cookies import extract_cookies_to_jar
from ._internal_utils import to_native_string
from .utils import parse_dict_header
def build_digest_header(self, method, url):
    """
        :rtype: str
        """
    realm = self._thread_local.chal['realm']
    nonce = self._thread_local.chal['nonce']
    qop = self._thread_local.chal.get('qop')
    algorithm = self._thread_local.chal.get('algorithm')
    opaque = self._thread_local.chal.get('opaque')
    hash_utf8 = None
    if algorithm is None:
        _algorithm = 'MD5'
    else:
        _algorithm = algorithm.upper()
    if _algorithm == 'MD5' or _algorithm == 'MD5-SESS':

        def md5_utf8(x):
            if isinstance(x, str):
                x = x.encode('utf-8')
            return hashlib.md5(x).hexdigest()
        hash_utf8 = md5_utf8
    elif _algorithm == 'SHA':

        def sha_utf8(x):
            if isinstance(x, str):
                x = x.encode('utf-8')
            return hashlib.sha1(x).hexdigest()
        hash_utf8 = sha_utf8
    elif _algorithm == 'SHA-256':

        def sha256_utf8(x):
            if isinstance(x, str):
                x = x.encode('utf-8')
            return hashlib.sha256(x).hexdigest()
        hash_utf8 = sha256_utf8
    elif _algorithm == 'SHA-512':

        def sha512_utf8(x):
            if isinstance(x, str):
                x = x.encode('utf-8')
            return hashlib.sha512(x).hexdigest()
        hash_utf8 = sha512_utf8
    KD = lambda s, d: hash_utf8('%s:%s' % (s, d))
    if hash_utf8 is None:
        return None
    entdig = None
    p_parsed = urlparse(url)
    path = p_parsed.path or '/'
    if p_parsed.query:
        path += '?' + p_parsed.query
    A1 = '%s:%s:%s' % (self.username, realm, self.password)
    A2 = '%s:%s' % (method, path)
    HA1 = hash_utf8(A1)
    HA2 = hash_utf8(A2)
    if nonce == self._thread_local.last_nonce:
        self._thread_local.nonce_count += 1
    else:
        self._thread_local.nonce_count = 1
    ncvalue = '%08x' % self._thread_local.nonce_count
    s = str(self._thread_local.nonce_count).encode('utf-8')
    s += nonce.encode('utf-8')
    s += time.ctime().encode('utf-8')
    s += os.urandom(8)
    cnonce = hashlib.sha1(s).hexdigest()[:16]
    if _algorithm == 'MD5-SESS':
        HA1 = hash_utf8('%s:%s:%s' % (HA1, nonce, cnonce))
    if not qop:
        respdig = KD(HA1, '%s:%s' % (nonce, HA2))
    elif qop == 'auth' or 'auth' in qop.split(','):
        noncebit = '%s:%s:%s:%s:%s' % (nonce, ncvalue, cnonce, 'auth', HA2)
        respdig = KD(HA1, noncebit)
    else:
        return None
    self._thread_local.last_nonce = nonce
    base = 'username="%s", realm="%s", nonce="%s", uri="%s", response="%s"' % (self.username, realm, nonce, path, respdig)
    if opaque:
        base += ', opaque="%s"' % opaque
    if algorithm:
        base += ', algorithm="%s"' % algorithm
    if entdig:
        base += ', digest="%s"' % entdig
    if qop:
        base += ', qop="auth", nc=%s, cnonce="%s"' % (ncvalue, cnonce)
    return 'Digest %s' % base