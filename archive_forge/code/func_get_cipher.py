from dissononce.extras.meta.cipher import MAP_CIPHER
def get_cipher(self, name):
    if name in MAP_CIPHER:
        return MAP_CIPHER[name]()
    raise ValueError('Unsupported Cipher: %s' % name)