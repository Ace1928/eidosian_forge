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
def dumptrailers(out: TextIO, doc: PDFDocument, show_fallback_xref: bool=False) -> None:
    for xref in doc.xrefs:
        if not isinstance(xref, PDFXRefFallback) or show_fallback_xref:
            out.write('<trailer>\n')
            dumpxml(out, xref.get_trailer())
            out.write('\n</trailer>\n\n')
    no_xrefs = all((isinstance(xref, PDFXRefFallback) for xref in doc.xrefs))
    if no_xrefs and (not show_fallback_xref):
        msg = 'This PDF does not have an xref. Use --show-fallback-xref if you want to display the content of a fallback xref that contains all objects.'
        logger.warning(msg)
    return