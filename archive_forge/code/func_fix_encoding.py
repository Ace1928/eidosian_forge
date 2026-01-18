from __future__ import annotations
import unicodedata
import warnings
from typing import (
from ftfy import bad_codecs
from ftfy import chardata, fixes
from ftfy.badness import is_bad
from ftfy.formatting import display_ljust
def fix_encoding(text: str, config: Optional[TextFixerConfig]=None, **kwargs):
    """
    Apply just the encoding-fixing steps of ftfy to this text. Returns the
    fixed text, discarding the explanation.

        >>> fix_encoding("Ã³")
        'ó'
        >>> fix_encoding("&ATILDE;&SUP3;")
        '&ATILDE;&SUP3;'
    """
    if config is None:
        config = TextFixerConfig(explain=False)
    config = _config_from_kwargs(config, kwargs)
    fixed, _explan = fix_encoding_and_explain(text, config)
    return fixed