import codecs
import sys
from future import utils
def decodefilename(fn):
    return fn.decode(FS_ENCODING, FS_ERRORS)