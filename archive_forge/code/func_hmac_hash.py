from dissononce.hash.hash import Hash
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.hmac import HMAC
def hmac_hash(self, key, data):
    hmac = HMAC(key=key, algorithm=hashes.SHA256(), backend=default_backend())
    hmac.update(data=data)
    return hmac.finalize()