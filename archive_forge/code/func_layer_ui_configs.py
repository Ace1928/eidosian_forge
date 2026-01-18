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
def layer_ui_configs(self):
    """Show OC visibility status modifiable by user."""
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    info = mupdf.PdfLayerConfigUi()
    n = mupdf.pdf_count_layer_config_ui(pdf)
    rc = []
    for i in range(n):
        mupdf.pdf_layer_config_ui_info(pdf, i, info)
        if info.type == 1:
            type_ = 'checkbox'
        elif info.type == 2:
            type_ = 'radiobox'
        else:
            type_ = 'label'
        item = {'number': i, 'text': info.text, 'depth': info.depth, 'type': type_, 'on': info.selected, 'locked': info.locked}
        rc.append(item)
    return rc