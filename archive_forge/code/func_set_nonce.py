def set_nonce(self, nonce):
    """
        SetNonce(nonce): Sets n = nonce.
        This function is used for handling out-of-order transport messages

        :param nonce:
        :type nonce: int
        :return:
        :rtype:
        """
    self._nonce = nonce