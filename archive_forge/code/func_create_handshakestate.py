from dissononce.dh.dh import DH
from dissononce.cipher.cipher import Cipher
from dissononce.hash.hash import Hash
from dissononce.processing.handshakepatterns.handshakepattern import HandshakePattern
from dissononce.processing.impl.handshakestate import HandshakeState
from dissononce.processing.impl.symmetricstate import SymmetricState
from dissononce.processing.impl.cipherstate import CipherState
def create_handshakestate(self, symmetricstate=None, dh=None):
    """
        :param symmetricstate:
        :type symmetricstate: SymmetricState
        :param dh:
        :type dh: DH
        :return:
        :rtype: HandshakeState
        """
    return HandshakeState(symmetricstate or self.create_symmetricstate(), dh or self._dh)