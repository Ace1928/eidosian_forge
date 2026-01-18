import pickle
from Crypto.Cipher import AES
from base64 import b64encode, b64decode
def create_cipher(key, seed):
    if len(seed) != 16:
        raise ValueError('Choose a seed of 16 bytes')
    if len(key) != 32:
        raise ValueError('Choose a key of 32 bytes')
    return AES.new(key, AES.MODE_CBC, seed)