def decrypt_with_ad(self, ad, ciphertext):
    """
        DecryptWithAd(ad, ciphertext):
        If k is non-empty returns DECRYPT(k, n++, ad, ciphertext). Otherwise returns ciphertext.
        If an authentication failure occurs in DECRYPT() then n is not incremented
        and an error is signaled to the caller.

        :param ad:
        :type ad: bytes
        :param ciphertext:
        :type ciphertext: bytes
        :return: bytes
        :rtype:
        """
    if self._key is None:
        return ciphertext
    result = self._cipher.decrypt(self._key, self._nonce, ad, ciphertext)
    self._nonce += 1
    return result