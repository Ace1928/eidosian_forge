from .err import OperationalError
from functools import partial
import hashlib
def caching_sha2_password_auth(conn, pkt):
    if not conn.password:
        return _roundtrip(conn, b'')
    if pkt.is_auth_switch_request():
        if DEBUG:
            print('caching sha2: Trying fast path')
        conn.salt = pkt.read_all()
        scrambled = scramble_caching_sha2(conn.password, conn.salt)
        pkt = _roundtrip(conn, scrambled)
    if not pkt.is_extra_auth_data():
        raise OperationalError('caching sha2: Unknown packet for fast auth: %s' % pkt._data[:1])
    pkt.advance(1)
    n = pkt.read_uint8()
    if n == 3:
        if DEBUG:
            print('caching sha2: succeeded by fast path.')
        pkt = conn._read_packet()
        pkt.check_error()
        return pkt
    if n != 4:
        raise OperationalError('caching sha2: Unknown result for fast auth: %s' % n)
    if DEBUG:
        print('caching sha2: Trying full auth...')
    if conn._secure:
        if DEBUG:
            print('caching sha2: Sending plain password via secure connection')
        return _roundtrip(conn, conn.password + b'\x00')
    if not conn.server_public_key:
        pkt = _roundtrip(conn, b'\x02')
        if not pkt.is_extra_auth_data():
            raise OperationalError('caching sha2: Unknown packet for public key: %s' % pkt._data[:1])
        conn.server_public_key = pkt._data[1:]
        if DEBUG:
            print(conn.server_public_key.decode('ascii'))
    data = sha2_rsa_encrypt(conn.password, conn.salt, conn.server_public_key)
    pkt = _roundtrip(conn, data)