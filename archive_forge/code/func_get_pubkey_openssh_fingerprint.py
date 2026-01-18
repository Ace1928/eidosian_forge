import hashlib
from libcloud.utils.py3 import b, hexadigits, base64_decode_string
def get_pubkey_openssh_fingerprint(pubkey):
    if not cryptography_available:
        raise RuntimeError('cryptography is not available')
    public_key = serialization.load_ssh_public_key(b(pubkey), backend=default_backend())
    pub_openssh = public_key.public_bytes(encoding=serialization.Encoding.OpenSSH, format=serialization.PublicFormat.OpenSSH)[7:]
    return _to_md5_fingerprint(base64_decode_string(pub_openssh))