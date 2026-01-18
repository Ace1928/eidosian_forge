import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def doc_join(args):
    """Join pages from several PDF documents."""
    doc_list = args.input
    doc = fitz.open()
    for src_item in doc_list:
        src_list = src_item.split(',')
        password = src_list[1] if len(src_list) > 1 else None
        src = open_file(src_list[0], password, pdf=True)
        pages = ','.join(src_list[2:])
        if pages:
            page_list = get_list(','.join(src_list[2:]), src.page_count + 1)
        else:
            page_list = range(1, src.page_count + 1)
        for i in page_list:
            doc.insert_pdf(src, from_page=i - 1, to_page=i - 1)
        src.close()
    doc.save(args.output, garbage=4, deflate=True)
    doc.close()