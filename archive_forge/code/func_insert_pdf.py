import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def insert_pdf(self, docsrc, from_page=-1, to_page=-1, start_at=-1, rotate=-1, links=1, annots=1, show_progress=0, final=1, _gmap=None):
    """Insert a page range from another PDF.

        Args:
            docsrc: PDF to copy from. Must be different object, but may be same file.
            from_page: (int) first source page to copy, 0-based, default 0.
            to_page: (int) last source page to copy, 0-based, default last page.
            start_at: (int) from_page will become this page number in target.
            rotate: (int) rotate copied pages, default -1 is no change.
            links: (int/bool) whether to also copy links.
            annots: (int/bool) whether to also copy annotations.
            show_progress: (int) progress message interval, 0 is no messages.
            final: (bool) indicates last insertion from this source PDF.
            _gmap: internal use only

        Copy sequence reversed if from_page > to_page."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    if self._graft_id == docsrc._graft_id:
        raise ValueError('source and target cannot be same object')
    sa = start_at
    if sa < 0:
        sa = self.page_count
    if len(docsrc) > show_progress > 0:
        inname = os.path.basename(docsrc.name)
        if not inname:
            inname = 'memory PDF'
        outname = os.path.basename(self.name)
        if not outname:
            outname = 'memory PDF'
        message("Inserting '%s' at '%s'" % (inname, outname))
    isrt = docsrc._graft_id
    _gmap = self.Graftmaps.get(isrt, None)
    if _gmap is None:
        _gmap = Graftmap(self)
        self.Graftmaps[isrt] = _gmap
    if g_use_extra:
        extra_FzDocument_insert_pdf(self.this, docsrc.this, from_page, to_page, start_at, rotate, links, annots, show_progress, final, _gmap)
    else:
        pdfout = _as_pdf_document(self)
        pdfsrc = _as_pdf_document(docsrc)
        outCount = mupdf.fz_count_pages(self)
        srcCount = mupdf.fz_count_pages(docsrc.this)
        fp = from_page
        tp = to_page
        sa = start_at
        fp = max(fp, 0)
        fp = min(fp, srcCount - 1)
        if tp < 0:
            tp = srcCount - 1
        tp = min(tp, srcCount - 1)
        if sa < 0:
            sa = outCount
        sa = min(sa, outCount)
        if not pdfout.m_internal or not pdfsrc.m_internal:
            raise TypeError('source or target not a PDF')
        ENSURE_OPERATION(pdfout)
        JM_merge_range(pdfout, pdfsrc, fp, tp, sa, rotate, links, annots, show_progress, _gmap)
    self._reset_page_refs()
    if links:
        self._do_links(docsrc, from_page=from_page, to_page=to_page, start_at=sa)
    if final == 1:
        self.Graftmaps[isrt] = None