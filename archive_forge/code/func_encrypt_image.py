from axolotl.kdf.hkdfv3 import HKDFv3
from axolotl.util.byteutil import ByteUtil
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import hmac
import hashlib
def encrypt_image(self, plaintext, ref_key):
    return self.encrypt(plaintext, ref_key, self.INFO_IMAGE)