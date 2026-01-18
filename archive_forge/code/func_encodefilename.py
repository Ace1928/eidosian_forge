import codecs
import sys
from future import utils
def encodefilename(fn):
    if FS_ENCODING == 'ascii':
        encoded = []
        for index, ch in enumerate(fn):
            code = ord(ch)
            if code < 128:
                ch = bytes_chr(code)
            elif 56448 <= code <= 56575:
                ch = bytes_chr(code - 56320)
            else:
                raise UnicodeEncodeError(FS_ENCODING, fn, index, index + 1, 'ordinal not in range(128)')
            encoded.append(ch)
        return bytes().join(encoded)
    elif FS_ENCODING == 'utf-8':
        encoded = []
        for index, ch in enumerate(fn):
            code = ord(ch)
            if 55296 <= code <= 57343:
                if 56448 <= code <= 56575:
                    ch = bytes_chr(code - 56320)
                    encoded.append(ch)
                else:
                    raise UnicodeEncodeError(FS_ENCODING, fn, index, index + 1, 'surrogates not allowed')
            else:
                ch_utf8 = ch.encode('utf-8')
                encoded.append(ch_utf8)
        return bytes().join(encoded)
    else:
        return fn.encode(FS_ENCODING, FS_ERRORS)