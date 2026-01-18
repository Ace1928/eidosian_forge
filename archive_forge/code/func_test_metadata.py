import array
import base64
import contextlib
import gc
import io
import math
import os
import shutil
import sys
import tempfile
import cairocffi
import pikepdf
import pytest
from . import (
@pytest.mark.xfail(cairo_version() < 11504, reason='Cairo version too low')
def test_metadata():
    file_obj = io.BytesIO()
    surface = PDFSurface(file_obj, 1, 1)
    surface.set_metadata(PDF_METADATA_TITLE, 'title')
    surface.set_metadata(PDF_METADATA_SUBJECT, 'subject')
    surface.set_metadata(PDF_METADATA_CREATOR, 'creator')
    surface.set_metadata(PDF_METADATA_AUTHOR, 'author')
    surface.set_metadata(PDF_METADATA_KEYWORDS, 'keywords')
    surface.set_metadata(PDF_METADATA_CREATE_DATE, '2013-07-21T23:46:00+01:00')
    surface.set_metadata(PDF_METADATA_MOD_DATE, '2013-07-21T23:46:00Z')
    surface.finish()
    pdf = pikepdf.Pdf.open(file_obj)
    assert pdf.docinfo['/Title'] == 'title'
    assert pdf.docinfo['/Subject'] == 'subject'
    assert pdf.docinfo['/Creator'] == 'creator'
    assert pdf.docinfo['/Author'] == 'author'
    assert pdf.docinfo['/Keywords'] == 'keywords'
    assert str(pdf.docinfo['/CreationDate']).startswith("20130721234600+01'00")
    assert pdf.docinfo['/ModDate'] == '20130721234600Z'