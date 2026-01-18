from base64 import encodebytes as _bencode
from quopri import encodestring as _encodestring
def encode_noop(msg):
    """Do nothing."""