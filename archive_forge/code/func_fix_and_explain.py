from __future__ import annotations
import unicodedata
import warnings
from typing import (
from ftfy import bad_codecs
from ftfy import chardata, fixes
from ftfy.badness import is_bad
from ftfy.formatting import display_ljust
def fix_and_explain(text: str, config: Optional[TextFixerConfig]=None, **kwargs) -> ExplainedText:
    """
    Fix text as a single segment, returning the fixed text and an explanation
    of what was fixed.

    The explanation is a list of steps that can be applied with
    :func:`apply_plan`, or if config.explain is False, it will be None.
    """
    if config is None:
        config = TextFixerConfig()
    if isinstance(text, bytes):
        raise UnicodeError(BYTES_ERROR_TEXT)
    config = _config_from_kwargs(config, kwargs)
    if config.unescape_html == 'auto' and '<' in text:
        config = config._replace(unescape_html=False)
    if config.explain:
        steps: Optional[List[ExplanationStep]] = []
    else:
        steps = None
    while True:
        origtext = text
        text = _try_fix('unescape_html', text, config, steps)
        if config.fix_encoding:
            if steps is None:
                text = fix_encoding(text)
            else:
                text, encoding_steps = fix_encoding_and_explain(text, config)
                if encoding_steps is not None:
                    steps.extend(encoding_steps)
        for fixer in ['fix_c1_controls', 'fix_latin_ligatures', 'fix_character_width', 'uncurl_quotes', 'fix_line_breaks', 'fix_surrogates', 'remove_terminal_escapes', 'remove_control_chars']:
            text = _try_fix(fixer, text, config, steps)
        if config.normalization is not None:
            fixed = unicodedata.normalize(config.normalization, text)
            if steps is not None and fixed != text:
                steps.append(ExplanationStep('normalize', config.normalization))
            text = fixed
        if text == origtext:
            return ExplainedText(text, steps)