from logging import getLogger as get_logger
from cryptography.hazmat.primitives.serialization import Encoding as _cryptography_encoding
import cryptography.x509 as _x509
def load_x509_certificate(data, cert_type='pem'):
    cert_reader = _x509_loaders.get(cert_type)
    if not cert_reader:
        cert_reader = _x509_loaders.get('pem')
        context = {'message': 'Unknown cert_type, falling back to default', 'cert_type': cert_type, 'default': DEFAULT_CERT_TYPE}
        logger.warning(context)
    cert = cert_reader(data)
    return cert