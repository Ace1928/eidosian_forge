import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def embedded_get(args):
    """Retrieve contents of an embedded file."""
    doc = open_file(args.input, args.password, pdf=True)
    exception_types = (ValueError, fitz.mupdf.FzErrorBase)
    if fitz.mupdf_version_tuple < (1, 24):
        exception_types = ValueError
    try:
        stream = doc.embfile_get(args.name)
        d = doc.embfile_info(args.name)
    except exception_types as e:
        sys.exit(f'no such embedded file {args.name!r}: {e}')
    filename = args.output if args.output else d['filename']
    with open(filename, 'wb') as output:
        output.write(stream)
    fitz.message("saved entry '%s' as '%s'" % (args.name, filename))
    doc.close()