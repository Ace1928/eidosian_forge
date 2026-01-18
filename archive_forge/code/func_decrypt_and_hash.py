from dissononce.processing.impl.cipherstate import CipherState
from dissononce.processing.symmetricstate import SymmetricState as BaseSymmetricState
def decrypt_and_hash(self, ciphertext):
    """
        DecryptAndHash(ciphertext):
        Sets plaintext = DecryptWithAd(h, ciphertext), calls MixHash(ciphertext), and returns plaintext.
        Note that if k is empty, the DecryptWithAd() call will set plaintext equal to ciphertext.

        :param ciphertext:
        :type ciphertext: bytes
        :return:
        :rtype: bytes
        """
    plaintext = self._cipherstate.decrypt_with_ad(self._h, ciphertext)
    self.mix_hash(ciphertext)
    return plaintext