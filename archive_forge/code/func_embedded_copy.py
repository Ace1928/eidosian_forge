import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def embedded_copy(args):
    """Copy embedded files between PDFs."""
    doc = open_file(args.input, args.password, pdf=True)
    if not doc.can_save_incrementally() and (not args.output or args.output == args.input):
        sys.exit('cannot save PDF incrementally')
    src = open_file(args.source, args.pwdsource)
    names = set(args.name) if args.name else set()
    src_names = set(src.embfile_names())
    if names:
        if not names <= src_names:
            sys.exit('not all names are contained in source')
    else:
        names = src_names
    if not names:
        sys.exit('nothing to copy')
    intersect = names & set(doc.embfile_names())
    if intersect:
        sys.exit('following names already exist in receiving PDF: %s' % str(intersect))
    for item in names:
        info = src.embfile_info(item)
        buff = src.embfile_get(item)
        doc.embfile_add(item, buff, filename=info['filename'], ufilename=info['ufilename'], desc=info['desc'])
        fitz.message("copied entry '%s' from '%s'" % (item, src.name))
    src.close()
    if args.output and args.output != args.input:
        doc.save(args.output, garbage=3)
    else:
        doc.saveIncr()
    doc.close()