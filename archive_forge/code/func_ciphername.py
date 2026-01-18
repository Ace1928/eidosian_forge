from dissononce.processing.impl.cipherstate import CipherState
from dissononce.processing.symmetricstate import SymmetricState as BaseSymmetricState
@property
def ciphername(self):
    return self._cipherstate.cipher.name