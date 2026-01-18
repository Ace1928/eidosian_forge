from __future__ import annotations
import array
import struct
from typing import TYPE_CHECKING
from babel.messages.catalog import Catalog, Message
def read_mo(fileobj: SupportsRead[bytes]) -> Catalog:
    """Read a binary MO file from the given file-like object and return a
    corresponding `Catalog` object.

    :param fileobj: the file-like object to read the MO file from

    :note: The implementation of this function is heavily based on the
           ``GNUTranslations._parse`` method of the ``gettext`` module in the
           standard library.
    """
    catalog = Catalog()
    headers = {}
    filename = getattr(fileobj, 'name', '')
    buf = fileobj.read()
    buflen = len(buf)
    unpack = struct.unpack
    magic = unpack('<I', buf[:4])[0]
    if magic == LE_MAGIC:
        version, msgcount, origidx, transidx = unpack('<4I', buf[4:20])
        ii = '<II'
    elif magic == BE_MAGIC:
        version, msgcount, origidx, transidx = unpack('>4I', buf[4:20])
        ii = '>II'
    else:
        raise OSError(0, 'Bad magic number', filename)
    for _i in range(msgcount):
        mlen, moff = unpack(ii, buf[origidx:origidx + 8])
        mend = moff + mlen
        tlen, toff = unpack(ii, buf[transidx:transidx + 8])
        tend = toff + tlen
        if mend < buflen and tend < buflen:
            msg = buf[moff:mend]
            tmsg = buf[toff:tend]
        else:
            raise OSError(0, 'File is corrupt', filename)
        if mlen == 0:
            lastkey = key = None
            for item in tmsg.splitlines():
                item = item.strip()
                if not item:
                    continue
                if b':' in item:
                    key, value = item.split(b':', 1)
                    lastkey = key = key.strip().lower()
                    headers[key] = value.strip()
                elif lastkey:
                    headers[lastkey] += b'\n' + item
        if b'\x04' in msg:
            ctxt, msg = msg.split(b'\x04')
        else:
            ctxt = None
        if b'\x00' in msg:
            msg = msg.split(b'\x00')
            tmsg = tmsg.split(b'\x00')
            if catalog.charset:
                msg = [x.decode(catalog.charset) for x in msg]
                tmsg = [x.decode(catalog.charset) for x in tmsg]
        elif catalog.charset:
            msg = msg.decode(catalog.charset)
            tmsg = tmsg.decode(catalog.charset)
        catalog[msg] = Message(msg, tmsg, context=ctxt)
        origidx += 8
        transidx += 8
    catalog.mime_headers = headers.items()
    return catalog