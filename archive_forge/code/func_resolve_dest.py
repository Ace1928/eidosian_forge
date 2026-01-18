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