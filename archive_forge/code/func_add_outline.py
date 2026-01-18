import ctypes
import io
import operator
import os
import sys
import weakref
from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile
from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontOptions, _encode_string
def add_outline(self, parent_id, utf8, link_attribs, flags=None):
    """Add an item to the document outline hierarchy.

        The outline has the ``utf8`` name and links to the location specified
        by ``link_attribs``. Link attributes have the same keys and values as
        the Link Tag, excluding the ``rect`` attribute. The item will be a
        child of the item with id ``parent_id``. Use ``PDF_OUTLINE_ROOT`` as
        the parent id of top level items.

        :param parent_id:
            the id of the parent item or ``PDF_OUTLINE_ROOT`` if this is a
            top level item.
        :param utf8: the name of the outline.
        :param link_attribs:
            the link attributes specifying where this outline links to.
        :param flags: outline item flags.

        :return: the id for the added item.

        *New in cairo 1.16.*

        *New in cairocffi 0.9.*

        """
    if flags is None:
        flags = 0
    value = cairo.cairo_pdf_surface_add_outline(self._pointer, parent_id, _encode_string(utf8), _encode_string(link_attribs), flags)
    self._check_status()
    return value