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
def resolve_names(self):
    """Convert the PDF's destination names into a Python dict.

        The only parameter is the fitz.Document.
        All names found in the catalog under keys "/Dests" and "/Names/Dests" are
        being included.

        Returns:
            A dcitionary with the following layout:
            - key: (str) the name
            - value: (dict) with the following layout:
                * "page":  target page number (0-based). If no page number found -1.
                * "to": (x, y) target point on page - currently in PDF coordinates,
                        i.e. point (0,0) is the bottom-left of the page.
                * "zoom": (float) the zoom factor
                * "dest": (str) only occurs if the target location on the page has
                        not been provided as "/XYZ" or if no page number was found.
            Examples:
            {'__bookmark_1': {'page': 0, 'to': (0.0, 541.0), 'zoom': 0.0},
            '__bookmark_2': {'page': 0, 'to': (0.0, 481.45), 'zoom': 0.0}}

            or

            '21154a7c20684ceb91f9c9adc3b677c40': {'page': -1, 'dest': '/XYZ 15.75 1486 0'}, ...
        """
    if hasattr(self, '_resolved_names'):
        return self._resolved_names
    page_xrefs = {self.page_xref(i): i for i in range(self.page_count)}

    def obj_string(obj):
        """Return string version of a PDF object definition."""
        buffer = mupdf.fz_new_buffer(512)
        output = mupdf.FzOutput(buffer)
        mupdf.pdf_print_obj(output, obj, 1, 0)
        output.fz_close_output()
        return JM_UnicodeFromBuffer(buffer)

    def get_array(val):
        """Generate value of one item of the names dictionary."""
        templ_dict = {'page': -1, 'dest': ''}
        if val.pdf_is_indirect():
            val = mupdf.pdf_resolve_indirect(val)
        if val.pdf_is_array():
            array = obj_string(val)
        elif val.pdf_is_dict():
            array = obj_string(mupdf.pdf_dict_gets(val, 'D'))
        else:
            return templ_dict
        array = array.replace('null', '0')[1:-1]
        idx = array.find('/')
        if idx < 1:
            templ_dict['dest'] = array
            return templ_dict
        subval = array[:idx]
        array = array[idx:]
        templ_dict['dest'] = array
        if array.startswith('/XYZ'):
            del templ_dict['dest']
            arr_t = array.split()[1:]
            x, y, z = tuple(map(float, arr_t))
            templ_dict['to'] = (x, y)
            templ_dict['zoom'] = z
        if '0 R' in subval:
            templ_dict['page'] = page_xrefs.get(int(subval.split()[0]), -1)
        else:
            templ_dict['page'] = int(subval)
        return templ_dict

    def fill_dict(dest_dict, pdf_dict):
        """Generate name resolution items for pdf_dict.

            This may be either "/Names/Dests" or just "/Dests"
            """
        name_count = mupdf.pdf_dict_len(pdf_dict)
        for i in range(name_count):
            key = mupdf.pdf_dict_get_key(pdf_dict, i)
            val = mupdf.pdf_dict_get_val(pdf_dict, i)
            if key.pdf_is_name():
                dict_key = key.pdf_to_name()
            else:
                message(f'key {i} is no /Name')
                dict_key = None
            if dict_key:
                dest_dict[dict_key] = get_array(val)
    pdf = mupdf.pdf_document_from_fz_document(self)
    catalog = mupdf.pdf_dict_gets(mupdf.pdf_trailer(pdf), 'Root')
    dest_dict = {}
    dests = mupdf.pdf_new_name('Dests')
    old_dests = mupdf.pdf_dict_get(catalog, dests)
    if old_dests.pdf_is_dict():
        fill_dict(dest_dict, old_dests)
    tree = mupdf.pdf_load_name_tree(pdf, dests)
    if tree.pdf_is_dict():
        fill_dict(dest_dict, tree)
    self._resolved_names = dest_dict
    return dest_dict