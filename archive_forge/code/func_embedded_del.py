import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def embedded_del(args):
    """Delete an embedded file entry."""
    doc = open_file(args.input, args.password, pdf=True)
    if not doc.can_save_incrementally() and (not args.output or args.output == args.input):
        sys.exit('cannot save PDF incrementally')
    exception_types = (ValueError, fitz.mupdf.FzErrorBase)
    if fitz.mupdf_version_tuple < (1, 24):
        exception_types = ValueError
    try:
        doc.embfile_del(args.name)
    except exception_types as e:
        sys.exit(f'no such embedded file {args.name!r}: {e}')
    if not args.output or args.output == args.input:
        doc.saveIncr()
    else:
        doc.save(args.output, garbage=1)
    doc.close()