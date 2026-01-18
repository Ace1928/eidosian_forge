import binascii
import hashlib
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import constant_time, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
from paramiko.message import Message
from paramiko.common import byte_chr
from paramiko.ssh_exception import SSHException
def parse_next(self, ptype, m):
    if self.transport.server_mode and ptype == _MSG_KEXECDH_INIT:
        return self._parse_kexecdh_init(m)
    elif not self.transport.server_mode and ptype == _MSG_KEXECDH_REPLY:
        return self._parse_kexecdh_reply(m)
    raise SSHException('KexCurve25519 asked to handle packet type {:d}'.format(ptype))