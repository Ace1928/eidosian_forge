from .err import OperationalError
from functools import partial
import hashlib
def sha256_password_auth(conn, pkt):
    if conn._secure:
        if DEBUG:
            print('sha256: Sending plain password')
        data = conn.password + b'\x00'
        return _roundtrip(conn, data)
    if pkt.is_auth_switch_request():
        conn.salt = pkt.read_all()
        if not conn.server_public_key and conn.password:
            if DEBUG:
                print('sha256: Requesting server public key')
            pkt = _roundtrip(conn, b'\x01')
    if pkt.is_extra_auth_data():
        conn.server_public_key = pkt._data[1:]
        if DEBUG:
            print('Received public key:\n', conn.server_public_key.decode('ascii'))
    if conn.password:
        if not conn.server_public_key:
            raise OperationalError("Couldn't receive server's public key")
        data = sha2_rsa_encrypt(conn.password, conn.salt, conn.server_public_key)
    else:
        data = b''
    return _roundtrip(conn, data)