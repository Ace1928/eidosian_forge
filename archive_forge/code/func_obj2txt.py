from __future__ import absolute_import, division, print_function
def obj2txt(openssl_lib, openssl_ffi, obj):
    buf_len = 80
    buf = openssl_ffi.new('char[]', buf_len)
    res = openssl_lib.OBJ_obj2txt(buf, buf_len, obj, 1)
    if res > buf_len - 1:
        buf_len = res + 1
        buf = openssl_ffi.new('char[]', buf_len)
        res = openssl_lib.OBJ_obj2txt(buf, buf_len, obj, 1)
    return openssl_ffi.buffer(buf, res)[:].decode()