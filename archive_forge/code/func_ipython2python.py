import base64
import os
import re
import textwrap
import warnings
from urllib.parse import quote
from xml.etree.ElementTree import Element
import bleach
from defusedxml import ElementTree  # type:ignore[import-untyped]
from nbconvert.preprocessors.sanitize import _get_default_css_sanitizer
def ipython2python(code):
    """Transform IPython syntax to pure Python syntax

    Parameters
    ----------
    code : str
        IPython code, to be transformed to pure Python
    """
    try:
        from IPython.core.inputtransformer2 import TransformerManager
    except ImportError:
        warnings.warn('IPython is needed to transform IPython syntax to pure Python. Install ipython if you need this functionality.', stacklevel=2)
        return code
    else:
        isp = TransformerManager()
        return isp.transform_cell(code)