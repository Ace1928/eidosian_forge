import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
def computeU(encryptionkey, encodestring=PadString, revision=None, documentId=None):
    revision = checkRevision(revision)
    from reportlab.lib.arciv import ArcIV
    if revision == 2:
        result = ArcIV(encryptionkey).encode(encodestring)
    elif revision == 3:
        assert documentId is not None, 'Revision 3 algorithm needs the document ID!'
        h = md5(PadString)
        h.update(rawBytes(documentId))
        tmp = h.digest()
        tmp = ArcIV(encryptionkey).encode(tmp)
        for n in range(1, 20):
            thisKey = xorKey(n, encryptionkey)
            tmp = ArcIV(thisKey).encode(tmp)
        while len(tmp) < 32:
            tmp += b'\x00'
        result = tmp
    if DEBUG:
        print('computeU(%s,%s,%s,%s)==>%s' % tuple([hexText(str(x)) for x in (encryptionkey, encodestring, revision, documentId, result)]))
    return result