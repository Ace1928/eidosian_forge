from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, c_size_t,
def oaep_decode(em, lHash, db):
    ret = _raw_pkcs1_decode.oaep_decode(c_uint8_ptr(em), c_size_t(len(em)), c_uint8_ptr(lHash), c_size_t(len(lHash)), c_uint8_ptr(db), c_size_t(len(db)))
    return ret