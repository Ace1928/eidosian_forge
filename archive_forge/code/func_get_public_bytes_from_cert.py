from logging import getLogger as get_logger
from cryptography.hazmat.primitives.serialization import Encoding as _cryptography_encoding
import cryptography.x509 as _x509
def get_public_bytes_from_cert(cert):
    data = cert.public_bytes(_cryptography_encoding.PEM).decode()
    return data