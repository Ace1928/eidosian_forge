from .decoder import JSONDecoder, JSONDecodeError
from .encoder import JSONEncoder
import codecs
def detect_encoding(b):
    bstartswith = b.startswith
    if bstartswith((codecs.BOM_UTF32_BE, codecs.BOM_UTF32_LE)):
        return 'utf-32'
    if bstartswith((codecs.BOM_UTF16_BE, codecs.BOM_UTF16_LE)):
        return 'utf-16'
    if bstartswith(codecs.BOM_UTF8):
        return 'utf-8-sig'
    if len(b) >= 4:
        if not b[0]:
            return 'utf-16-be' if b[1] else 'utf-32-be'
        if not b[1]:
            return 'utf-16-le' if b[2] or b[3] else 'utf-32-le'
    elif len(b) == 2:
        if not b[0]:
            return 'utf-16-be'
        if not b[1]:
            return 'utf-16-le'
    return 'utf-8'