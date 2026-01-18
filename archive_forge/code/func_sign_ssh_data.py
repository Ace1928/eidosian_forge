import bcrypt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher
import nacl.signing
from paramiko.message import Message
from paramiko.pkey import PKey, OPENSSH_AUTH_MAGIC, _unpad_openssh
from paramiko.util import b
from paramiko.ssh_exception import SSHException, PasswordRequiredException
def sign_ssh_data(self, data, algorithm=None):
    m = Message()
    m.add_string(self.name)
    m.add_string(self._signing_key.sign(data).signature)
    return m