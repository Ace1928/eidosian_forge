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
def page_merge(doc_des, doc_src, page_from, page_to, rotate, links, copy_annots, graft_map):
    """
    Deep-copies a source page to the target.
    Modified version of function of pdfmerge.c: we also copy annotations, but
    we skip some subtypes. In addition we rotate output.
    """
    if g_use_extra:
        return extra.page_merge(doc_des, doc_src, page_from, page_to, rotate, links, copy_annots, graft_map)
    known_page_objs = [PDF_NAME('Contents'), PDF_NAME('Resources'), PDF_NAME('MediaBox'), PDF_NAME('CropBox'), PDF_NAME('BleedBox'), PDF_NAME('TrimBox'), PDF_NAME('ArtBox'), PDF_NAME('Rotate'), PDF_NAME('UserUnit')]
    page_ref = mupdf.pdf_lookup_page_obj(doc_src, page_from)
    page_dict = mupdf.pdf_new_dict(doc_des, 4)
    mupdf.pdf_dict_put(page_dict, PDF_NAME('Type'), PDF_NAME('Page'))
    for i in range(len(known_page_objs)):
        obj = mupdf.pdf_dict_get_inheritable(page_ref, known_page_objs[i])
        if obj.m_internal:
            mupdf.pdf_dict_put(page_dict, known_page_objs[i], mupdf.pdf_graft_mapped_object(graft_map.this, obj))
    if copy_annots:
        old_annots = mupdf.pdf_dict_get(page_ref, PDF_NAME('Annots'))
        n = mupdf.pdf_array_len(old_annots)
        if n > 0:
            new_annots = mupdf.pdf_dict_put_array(page_dict, PDF_NAME('Annots'), n)
            for i in range(n):
                o = mupdf.pdf_array_get(old_annots, i)
                if not o.m_internal or mupdf.pdf_is_dict(o):
                    continue
                if mupdf.pdf_dict_gets(o, 'IRT').m_internal:
                    continue
                subtype = mupdf.pdf_dict_get(o, PDF_NAME('Subtype'))
                if mupdf.pdf_name_eq(subtype, PDF_NAME('Link')):
                    continue
                if mupdf.pdf_name_eq(subtype, PDF_NAME('Popup')):
                    continue
                if mupdf.pdf_name_eq(subtype, PDF_NAME('Widget')):
                    mupdf.fz_warn('skipping widget annotation')
                    continue
                if mupdf.pdf_name_eq(subtype, PDF_NAME('Widget')):
                    continue
                mupdf.pdf_dict_del(o, PDF_NAME('Popup'))
                mupdf.pdf_dict_del(o, PDF_NAME('P'))
                copy_o = mupdf.pdf_graft_mapped_object(graft_map.this, o)
                annot = mupdf.pdf_new_indirect(doc_des, mupdf.pdf_to_num(copy_o), 0)
                mupdf.pdf_array_push(new_annots, annot)
    if rotate != -1:
        mupdf.pdf_dict_put_int(page_dict, PDF_NAME('Rotate'), rotate)
    ref = mupdf.pdf_add_object(doc_des, page_dict)
    mupdf.pdf_insert_page(doc_des, page_to, ref)