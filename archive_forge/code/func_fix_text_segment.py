from __future__ import annotations
import unicodedata
import warnings
from typing import (
from ftfy import bad_codecs
from ftfy import chardata, fixes
from ftfy.badness import is_bad
from ftfy.formatting import display_ljust
def fix_text_segment(text: str, config: Optional[TextFixerConfig]=None, **kwargs):
    """
    Fix text as a single segment, with a consistent sequence of steps that
    are applied to fix the text. Discard the explanation.
    """
    if config is None:
        config = TextFixerConfig(explain=False)
    config = _config_from_kwargs(config, kwargs)
    fixed, _explan = fix_and_explain(text, config)
    return fixed