from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, c_size_t,
def pkcs1_decode(em, sentinel, expected_pt_len, output):
    if len(em) != len(output):
        raise ValueError('Incorrect output length')
    ret = _raw_pkcs1_decode.pkcs1_decode(c_uint8_ptr(em), c_size_t(len(em)), c_uint8_ptr(sentinel), c_size_t(len(sentinel)), c_size_t(expected_pt_len), c_uint8_ptr(output))
    return ret