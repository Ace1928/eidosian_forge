from paramiko.common import max_byte, zero_byte, byte_ord, byte_chr
import paramiko.util as util
from paramiko.util import b
from paramiko.sftp import int64
def encode_tlv(self, ident, val):
    self.content += byte_chr(ident)
    if len(val) > 127:
        lenstr = util.deflate_long(len(val))
        self.content += byte_chr(128 + len(lenstr)) + lenstr
    else:
        self.content += byte_chr(len(val))
    self.content += val