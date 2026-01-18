from paramiko.common import max_byte, zero_byte, byte_ord, byte_chr
import paramiko.util as util
from paramiko.util import b
from paramiko.sftp import int64
@staticmethod
def encode_sequence(data):
    ber = BER()
    for item in data:
        ber.encode(item)
    return ber.asbytes()