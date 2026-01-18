from .publickey import PublicKey
from .privatekey import PrivateKey
from dissononce.dh.x25519.x25519 import X25519DH
from dissononce.dh.x25519.keypair import KeyPair as X25519KeyPair
@property
def private(self):
    return self._private