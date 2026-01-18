from typing import Any, Optional
import OpenSSL._util as pyOpenSSLutil
import OpenSSL.SSL
import OpenSSL.version
from OpenSSL.crypto import X509Name
from scrapy.utils.python import to_unicode
def get_temp_key_info(ssl_object: Any) -> Optional[str]:
    if not hasattr(pyOpenSSLutil.lib, 'SSL_get_server_tmp_key'):
        return None
    temp_key_p = pyOpenSSLutil.ffi.new('EVP_PKEY **')
    if not pyOpenSSLutil.lib.SSL_get_server_tmp_key(ssl_object, temp_key_p):
        return None
    temp_key = temp_key_p[0]
    if temp_key == pyOpenSSLutil.ffi.NULL:
        return None
    temp_key = pyOpenSSLutil.ffi.gc(temp_key, pyOpenSSLutil.lib.EVP_PKEY_free)
    key_info = []
    key_type = pyOpenSSLutil.lib.EVP_PKEY_id(temp_key)
    if key_type == pyOpenSSLutil.lib.EVP_PKEY_RSA:
        key_info.append('RSA')
    elif key_type == pyOpenSSLutil.lib.EVP_PKEY_DH:
        key_info.append('DH')
    elif key_type == pyOpenSSLutil.lib.EVP_PKEY_EC:
        key_info.append('ECDH')
        ec_key = pyOpenSSLutil.lib.EVP_PKEY_get1_EC_KEY(temp_key)
        ec_key = pyOpenSSLutil.ffi.gc(ec_key, pyOpenSSLutil.lib.EC_KEY_free)
        nid = pyOpenSSLutil.lib.EC_GROUP_get_curve_name(pyOpenSSLutil.lib.EC_KEY_get0_group(ec_key))
        cname = pyOpenSSLutil.lib.EC_curve_nid2nist(nid)
        if cname == pyOpenSSLutil.ffi.NULL:
            cname = pyOpenSSLutil.lib.OBJ_nid2sn(nid)
        key_info.append(ffi_buf_to_string(cname))
    else:
        key_info.append(ffi_buf_to_string(pyOpenSSLutil.lib.OBJ_nid2sn(key_type)))
    key_info.append(f'{pyOpenSSLutil.lib.EVP_PKEY_bits(temp_key)} bits')
    return ', '.join(key_info)