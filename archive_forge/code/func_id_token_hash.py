import base64
import hashlib
import logging
import time
from json import loads
from oauthlib.oauth2.rfc6749.errors import (
def id_token_hash(self, value, hashfunc=hashlib.sha256):
    """
        Its value is the base64url encoding of the left-most half of the
        hash of the octets of the ASCII representation of the access_token
        value, where the hash algorithm used is the hash algorithm used in
        the alg Header Parameter of the ID Token's JOSE Header.

        For instance, if the alg is RS256, hash the access_token value
        with SHA-256, then take the left-most 128 bits and
        base64url-encode them.
        For instance, if the alg is HS512, hash the code value with
        SHA-512, then take the left-most 256 bits and base64url-encode
        them. The c_hash value is a case-sensitive string.

        Example of hash from OIDC specification (bound to a JWS using RS256):

        code:
        Qcb0Orv1zh30vL1MPRsbm-diHiMwcLyZvn1arpZv-Jxf_11jnpEX3Tgfvk

        c_hash:
        LDktKdoQak3Pk0cnXxCltA
        """
    digest = hashfunc(value.encode()).digest()
    left_most = len(digest) // 2
    return base64.urlsafe_b64encode(digest[:left_most]).decode().rstrip('=')