from dissononce.processing.impl.cipherstate import CipherState
from dissononce.processing.symmetricstate import SymmetricState as BaseSymmetricState
def mix_hash(self, data):
    """
        MixHash(data):
        Sets h = HASH(h || data).

        :param data:
        :type data: bytes
        :return:
        :rtype:
        """
    self._h = self._hashfn.hash(self._h + data)