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
def fullcopy_page(self, pno, to=-1):
    """Make a full page duplicate."""
    pdf = _as_pdf_document(self)
    page_count = mupdf.pdf_count_pages(pdf)
    try:
        ASSERT_PDF(pdf)
        if not _INRANGE(pno, 0, page_count - 1) or not _INRANGE(to, -1, page_count - 1):
            raise ValueError(MSG_BAD_PAGENO)
        page1 = mupdf.pdf_resolve_indirect(mupdf.pdf_lookup_page_obj(pdf, pno))
        page2 = mupdf.pdf_deep_copy_obj(page1)
        old_annots = mupdf.pdf_dict_get(page2, PDF_NAME('Annots'))
        if old_annots.m_internal:
            n = mupdf.pdf_array_len(old_annots)
            new_annots = mupdf.pdf_new_array(pdf, n)
            for i in range(n):
                o = mupdf.pdf_array_get(old_annots, i)
                subtype = mupdf.pdf_dict_get(o, PDF_NAME('Subtype'))
                if mupdf.pdf_name_eq(subtype, PDF_NAME('Popup')):
                    continue
                if mupdf.pdf_dict_gets(o, 'IRT').m_internal:
                    continue
                copy_o = mupdf.pdf_deep_copy_obj(mupdf.pdf_resolve_indirect(o))
                xref = mupdf.pdf_create_object(pdf)
                mupdf.pdf_update_object(pdf, xref, copy_o)
                copy_o = mupdf.pdf_new_indirect(pdf, xref, 0)
                mupdf.pdf_dict_del(copy_o, PDF_NAME('Popup'))
                mupdf.pdf_dict_del(copy_o, PDF_NAME('P'))
                mupdf.pdf_array_push(new_annots, copy_o)
            mupdf.pdf_dict_put(page2, PDF_NAME('Annots'), new_annots)
        res = JM_read_contents(page1)
        if res.m_internal:
            contents = mupdf.pdf_add_stream(pdf, mupdf.fz_new_buffer_from_copied_data(b' '), mupdf.PdfObj(), 0)
            JM_update_stream(pdf, contents, res, 1)
            mupdf.pdf_dict_put(page2, PDF_NAME('Contents'), contents)
        xref = mupdf.pdf_create_object(pdf)
        mupdf.pdf_update_object(pdf, xref, page2)
        page2 = mupdf.pdf_new_indirect(pdf, xref, 0)
        mupdf.pdf_insert_page(pdf, to, page2)
    finally:
        mupdf.ll_pdf_drop_page_tree(pdf.m_internal)
    self._reset_page_refs()