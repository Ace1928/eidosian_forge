import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
def encodePDF(key, objectNumber, generationNumber, string, revision=None):
    """Encodes a string or stream"""
    revision = checkRevision(revision)
    if revision in (2, 3):
        newkey = key
        n = objectNumber
        for i in range(3):
            newkey += int2Byte(n & 255)
            n = n >> 8
        n = generationNumber
        for i in range(2):
            newkey += int2Byte(n & 255)
            n = n >> 8
        md5output = md5(newkey).digest()
        if revision == 2:
            key = md5output[:10]
        elif revision == 3:
            key = md5output
        from reportlab.lib.arciv import ArcIV
        encrypted = ArcIV(key).encode(string)
    elif revision == 5:
        iv = os_urandom(16)
        encrypter = pyaes.Encrypter(pyaes.AESModeOfOperationCBC(key, iv=iv))
        string_len = len(string)
        padding = ''
        padding_len = 16 - string_len % 16 if string_len > 16 else 16 - string_len
        if padding_len > 0:
            padding = chr(padding_len) * padding_len
        if isinstance(string, str):
            string = (string + padding).encode('utf-8')
        else:
            string += asBytes(padding)
        encrypted = iv + encrypter.feed(string)
        encrypted += encrypter.feed()
    if DEBUG:
        print('encodePDF(%s,%s,%s,%s,%s)==>%s' % tuple([hexText(str(x)) for x in (key, objectNumber, generationNumber, string, revision, encrypted)]))
    return encrypted