from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
from paramiko.common import four_byte
from paramiko.message import Message
from paramiko.pkey import PKey
from paramiko.ssh_exception import SSHException
from paramiko.util import deflate_long
def get_by_curve_class(self, curve_class):
    for curve in self.ecdsa_curves:
        if curve.curve_class == curve_class:
            return curve