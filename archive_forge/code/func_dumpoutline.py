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
def dumpoutline(outfp: TextIO, fname: str, objids: Any, pagenos: Container[int], password: str='', dumpall: bool=False, codec: Optional[str]=None, extractdir: Optional[str]=None) -> None:
    fp = open(fname, 'rb')
    parser = PDFParser(fp)
    doc = PDFDocument(parser, password)
    pages = {page.pageid: pageno for pageno, page in enumerate(PDFPage.create_pages(doc), 1)}

    def resolve_dest(dest: object) -> Any:
        if isinstance(dest, (str, bytes)):
            dest = resolve1(doc.get_dest(dest))
        elif isinstance(dest, PSLiteral):
            dest = resolve1(doc.get_dest(dest.name))
        if isinstance(dest, dict):
            dest = dest['D']
        if isinstance(dest, PDFObjRef):
            dest = dest.resolve()
        return dest
    try:
        outlines = doc.get_outlines()
        outfp.write('<outlines>\n')
        for level, title, dest, a, se in outlines:
            pageno = None
            if dest:
                dest = resolve_dest(dest)
                pageno = pages[dest[0].objid]
            elif a:
                action = a
                if isinstance(action, dict):
                    subtype = action.get('S')
                    if subtype and repr(subtype) == "/'GoTo'" and action.get('D'):
                        dest = resolve_dest(action['D'])
                        pageno = pages[dest[0].objid]
            s = escape(title)
            outfp.write('<outline level="{!r}" title="{}">\n'.format(level, s))
            if dest is not None:
                outfp.write('<dest>')
                dumpxml(outfp, dest)
                outfp.write('</dest>\n')
            if pageno is not None:
                outfp.write('<pageno>%r</pageno>\n' % pageno)
            outfp.write('</outline>\n')
        outfp.write('</outlines>\n')
    except PDFNoOutlines:
        pass
    parser.close()
    fp.close()
    return