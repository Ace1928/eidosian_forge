from __future__ import annotations
import unicodedata
import warnings
from typing import (
from ftfy import bad_codecs
from ftfy import chardata, fixes
from ftfy.badness import is_bad
from ftfy.formatting import display_ljust
def fix_file(input_file: TextIO | BinaryIO, encoding: Optional[str]=None, config: Optional[TextFixerConfig]=None, **kwargs) -> Iterator[str]:
    """
    Fix text that is found in a file.

    If the file is being read as Unicode text, use that. If it's being read as
    bytes, then we hope an encoding was supplied. If not, unfortunately, we
    have to guess what encoding it is. We'll try a few common encodings, but we
    make no promises. See the `guess_bytes` function for how this is done.

    The output is a stream of fixed lines of text.
    """
    if config is None:
        config = TextFixerConfig()
    config = _config_from_kwargs(config, kwargs)
    for line in input_file:
        if isinstance(line, bytes):
            if encoding is None:
                line, encoding = guess_bytes(line)
            else:
                line = line.decode(encoding)
        if config.unescape_html == 'auto' and '<' in line:
            config = config._replace(unescape_html=False)
        fixed_line, _explan = fix_and_explain(line, config)
        yield fixed_line