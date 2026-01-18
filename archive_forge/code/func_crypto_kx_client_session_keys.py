from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_kx_client_session_keys(client_public_key: bytes, client_secret_key: bytes, server_public_key: bytes) -> Tuple[bytes, bytes]:
    """
    Generate session keys for the client.
    :param client_public_key:
    :type client_public_key: bytes
    :param client_secret_key:
    :type client_secret_key: bytes
    :param server_public_key:
    :type server_public_key: bytes
    :return: (rx_key, tx_key)
    :rtype: (bytes, bytes)
    """
    ensure(isinstance(client_public_key, bytes) and len(client_public_key) == crypto_kx_PUBLIC_KEY_BYTES, 'Client public key must be a {} bytes long bytes sequence'.format(crypto_kx_PUBLIC_KEY_BYTES), raising=exc.TypeError)
    ensure(isinstance(client_secret_key, bytes) and len(client_secret_key) == crypto_kx_SECRET_KEY_BYTES, 'Client secret key must be a {} bytes long bytes sequence'.format(crypto_kx_PUBLIC_KEY_BYTES), raising=exc.TypeError)
    ensure(isinstance(server_public_key, bytes) and len(server_public_key) == crypto_kx_PUBLIC_KEY_BYTES, 'Server public key must be a {} bytes long bytes sequence'.format(crypto_kx_PUBLIC_KEY_BYTES), raising=exc.TypeError)
    rx_key = ffi.new('unsigned char[]', crypto_kx_SESSION_KEY_BYTES)
    tx_key = ffi.new('unsigned char[]', crypto_kx_SESSION_KEY_BYTES)
    res = lib.crypto_kx_client_session_keys(rx_key, tx_key, client_public_key, client_secret_key, server_public_key)
    ensure(res == 0, 'Client session key generation failed.', raising=exc.CryptoError)
    return (ffi.buffer(rx_key, crypto_kx_SESSION_KEY_BYTES)[:], ffi.buffer(tx_key, crypto_kx_SESSION_KEY_BYTES)[:])