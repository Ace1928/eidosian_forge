import logging
import os.path
import re
import sys
from typing import Any, Container, Dict, Iterable, List, Optional, TextIO, Union, cast
from argparse import ArgumentParser
import pdfminer
from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines, PDFXRefFallback
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import PDFObjectNotFound, PDFValueError
from pdfminer.pdftypes import PDFStream, PDFObjRef, resolve1, stream_value
from pdfminer.psparser import PSKeyword, PSLiteral, LIT
from pdfminer.utils import isnumber
def extractembedded(fname: str, password: str, extractdir: str) -> None:

    def extract1(objid: int, obj: Dict[str, Any]) -> None:
        filename = os.path.basename(obj.get('UF') or cast(bytes, obj.get('F')).decode())
        fileref = obj['EF'].get('UF') or obj['EF'].get('F')
        fileobj = doc.getobj(fileref.objid)
        if not isinstance(fileobj, PDFStream):
            error_msg = 'unable to process PDF: reference for %r is not a PDFStream' % filename
            raise PDFValueError(error_msg)
        if fileobj.get('Type') is not LITERAL_EMBEDDEDFILE:
            raise PDFValueError('unable to process PDF: reference for %r is not an EmbeddedFile' % filename)
        path = os.path.join(extractdir, '%.6d-%s' % (objid, filename))
        if os.path.exists(path):
            raise IOError('file exists: %r' % path)
        print('extracting: %r' % path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = open(path, 'wb')
        out.write(fileobj.get_data())
        out.close()
        return
    with open(fname, 'rb') as fp:
        parser = PDFParser(fp)
        doc = PDFDocument(parser, password)
        extracted_objids = set()
        for xref in doc.xrefs:
            for objid in xref.get_objids():
                obj = doc.getobj(objid)
                if objid not in extracted_objids and isinstance(obj, dict) and (obj.get('Type') is LITERAL_FILESPEC):
                    extracted_objids.add(objid)
                    extract1(objid, obj)
    return