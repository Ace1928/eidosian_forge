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
def set_layer_ui_config(self, number, action=0):
    """Set / unset OC intent configuration."""
    if isinstance(number, str):
        select = [ui['number'] for ui in self.layer_ui_configs() if ui['text'] == number]
        if select == []:
            raise ValueError(f"bad OCG '{number}'.")
        number = select[0]
    pdf = _as_pdf_document(self)
    assert pdf
    if action == 1:
        mupdf.pdf_toggle_layer_config_ui(pdf, number)
    elif action == 2:
        mupdf.pdf_deselect_layer_config_ui(pdf, number)
    else:
        mupdf.pdf_select_layer_config_ui(pdf, number)