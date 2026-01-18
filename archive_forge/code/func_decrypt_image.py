from axolotl.kdf.hkdfv3 import HKDFv3
from axolotl.util.byteutil import ByteUtil
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import hmac
import hashlib
def decrypt_image(self, ciphertext, ref_key):
    return self.decrypt(ciphertext, ref_key, self.INFO_IMAGE)