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
@staticmethod
def add_pdf_links(document_or_stream, positions):
    """
        Adds links to PDF document.
        Args:
            document_or_stream:
                A PDF `Document` or raw PDF content, for example an
                `io.BytesIO` instance.
            positions:
                List of `ElementPosition`'s for `document_or_stream`,
                typically from Story.element_positions(). We raise an
                exception if two or more positions have same id.
        Returns:
            `document_or_stream` if a `Document` instance, otherwise a
            new `Document` instance.
        We raise an exception if an `href` in `positions` refers to an
        internal position `#<name>` but no item in `postions` has `id =
        name`.
        """
    if isinstance(document_or_stream, Document):
        document = document_or_stream
    else:
        document = Document('pdf', document_or_stream)
    id_to_position = dict()
    for position in positions:
        if position.open_close & 1 and position.id:
            if position.id in id_to_position:
                pass
            else:
                id_to_position[position.id] = position
    for position_from in positions:
        if position_from.open_close & 1 and position_from.href:
            link = dict()
            link['from'] = Rect(position_from.rect)
            if position_from.href.startswith('#'):
                target_id = position_from.href[1:]
                try:
                    position_to = id_to_position[target_id]
                except Exception as e:
                    if g_exceptions_verbose > 1:
                        exception_info()
                    raise RuntimeError(f'No destination with id={target_id}, required by position_from: {position_from}') from e
                if 0:
                    log(f'add_pdf_links(): making link from:')
                    log(f'add_pdf_links():    {position_from}')
                    log(f'add_pdf_links(): to:')
                    log(f'add_pdf_links():    {position_to}')
                link['kind'] = LINK_GOTO
                x0, y0, x1, y1 = position_to.rect
                link['to'] = Point(x0, y0)
                link['page'] = position_to.page_num - 1
            elif position_from.href.startswith('name:'):
                link['kind'] = LINK_NAMED
                link['name'] = position_from.href[5:]
            else:
                link['kind'] = LINK_URI
                link['uri'] = position_from.href
            document[position_from.page_num - 1].insert_link(link)
    return document