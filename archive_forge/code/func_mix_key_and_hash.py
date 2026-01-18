from dissononce.processing.impl.cipherstate import CipherState
from dissononce.processing.symmetricstate import SymmetricState as BaseSymmetricState
def mix_key_and_hash(self, input_key_material):
    """
        This function is used for handling pre-shared symmetric keys. It executes the following steps:

        Sets ck, temp_h, temp_k = HKDF(ck, input_key_material, 3).
        Calls MixHash(temp_h).
        If HASHLEN is 64, then truncates temp_k to 32 bytes.
        Calls InitializeKey(temp_k).

        :param input_key_material:
        :type input_key_material: bytes
        :return:
        :rtype:
        """
    self._ck, temp_h, temp_k = self._hashfn.hkdf(self._ck, input_key_material, 3)
    self.mix_hash(temp_h)
    if self._hashfn.hashlen == 64:
        temp_k = temp_k[:32]
    self._cipherstate.initialize_key(temp_k)