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
@pytest.mark.xfail(cairo_version() < 11510, reason='Cairo version too low')
def test_document_unit():
    surface = SVGSurface(None, 1, 2)
    assert surface.get_document_unit() in (SVG_UNIT_USER, SVG_UNIT_PT)
    file_obj = io.BytesIO()
    surface = SVGSurface(file_obj, 1, 2)
    surface.set_document_unit(SVG_UNIT_PX)
    assert surface.get_document_unit() == SVG_UNIT_PX
    surface.finish()
    pdf_bytes = file_obj.getvalue()
    assert b'width="1px"' in pdf_bytes
    assert b'height="2px"' in pdf_bytes
    file_obj = io.BytesIO()
    surface = SVGSurface(file_obj, 1, 2)
    surface.set_document_unit(SVG_UNIT_PC)
    assert surface.get_document_unit() == SVG_UNIT_PC
    surface.finish()
    pdf_bytes = file_obj.getvalue()
    assert b'width="1pc"' in pdf_bytes
    assert b'height="2pc"' in pdf_bytes