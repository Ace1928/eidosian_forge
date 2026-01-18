from kombu.serialization import dumps, loads, registry
from kombu.utils.encoding import bytes_to_str, ensure_bytes, str_to_bytes
from celery.app.defaults import DEFAULT_SECURITY_DIGEST
from celery.utils.serialization import b64decode, b64encode
from .certificate import Certificate, FSCertStore
from .key import PrivateKey
from .utils import get_digest_algorithm, reraise_errors
def register_auth(key=None, key_password=None, cert=None, store=None, digest=DEFAULT_SECURITY_DIGEST, serializer='json'):
    """Register security serializer."""
    s = SecureSerializer(key and PrivateKey(key, password=key_password), cert and Certificate(cert), store and FSCertStore(store), digest, serializer=serializer)
    registry.register('auth', s.serialize, s.deserialize, content_type='application/data', content_encoding='utf-8')