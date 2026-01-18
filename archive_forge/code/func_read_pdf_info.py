from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def read_pdf_info(self):
    self.file_size_total = len(self.buf)
    self.file_size_this = self.file_size_total - self.start_offset
    self.read_trailer()
    self.root_ref = self.trailer_dict[b'Root']
    self.info_ref = self.trailer_dict.get(b'Info', None)
    self.root = PdfDict(self.read_indirect(self.root_ref))
    if self.info_ref is None:
        self.info = PdfDict()
    else:
        self.info = PdfDict(self.read_indirect(self.info_ref))
    check_format_condition(b'Type' in self.root, '/Type missing in Root')
    check_format_condition(self.root[b'Type'] == b'Catalog', '/Type in Root is not /Catalog')
    check_format_condition(b'Pages' in self.root, '/Pages missing in Root')
    check_format_condition(isinstance(self.root[b'Pages'], IndirectReference), '/Pages in Root is not an indirect reference')
    self.pages_ref = self.root[b'Pages']
    self.page_tree_root = self.read_indirect(self.pages_ref)
    self.pages = self.linearize_page_tree(self.page_tree_root)
    self.orig_pages = self.pages[:]