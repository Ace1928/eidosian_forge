from __future__ import annotations
import os
import pickle
import types
from os import path
from typing import Any
from sphinx.application import ENV_PICKLE_FILENAME, Sphinx
from sphinx.builders.html import BuildInfo, StandaloneHTMLBuilder
from sphinx.locale import get_translation
from sphinx.util.osutil import SEP, copyfile, ensuredir, os_path
from sphinxcontrib.serializinghtml import jsonimpl
def get_target_uri(self, docname: str, typ: str | None=None) -> str:
    if docname == 'index':
        return ''
    if docname.endswith(SEP + 'index'):
        return docname[:-5]
    return docname + SEP