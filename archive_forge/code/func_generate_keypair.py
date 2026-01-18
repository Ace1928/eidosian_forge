from dissononce.dh.dh import DH
def generate_keypair(self, privatekey=None):
    return self._dh.generate_keypair(privatekey or self._privatekey)