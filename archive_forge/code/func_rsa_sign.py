from __future__ import print_function
import base64
import hashlib
import os
from cStringIO import StringIO
from M2Crypto import BIO, EVP, RSA, X509, m2
def rsa_sign(xml, ref_uri, private_key, password=None, cert=None, c14n_exc=True, sign_template=SIGN_REF_TMPL, key_info_template=KEY_INFO_RSA_TMPL):
    """Sign an XML document usign RSA (templates: enveloped -ref- or enveloping)"""
    ref_xml = canonicalize(xml, c14n_exc)
    signed_info = sign_template % {'ref_uri': ref_uri, 'digest_value': sha1_hash_digest(ref_xml)}
    signed_info = canonicalize(signed_info, c14n_exc)
    pkey = RSA.load_key(private_key, lambda *args, **kwargs: password)
    signature = pkey.sign(hashlib.sha1(signed_info).digest())
    return {'ref_xml': ref_xml, 'ref_uri': ref_uri, 'signed_info': signed_info, 'signature_value': base64.b64encode(signature), 'key_info': key_info(pkey, cert, key_info_template)}