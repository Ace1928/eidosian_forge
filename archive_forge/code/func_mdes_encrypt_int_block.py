from warnings import warn
from passlib.utils.decor import deprecated_function
from passlib.crypto.des import expand_des_key, des_encrypt_block, des_encrypt_int_block
import struct
@deprecated_function(deprecated='1.6', removed='1.8', replacement='passlib.crypto.des.des_encrypt_int_block()')
def mdes_encrypt_int_block(key, input, salt=0, rounds=1):
    if isinstance(key, bytes):
        if len(key) == 7:
            key = expand_des_key(key)
        key = _unpack_uint64(key)[0]
    return des_encrypt_int_block(key, input, salt, rounds)