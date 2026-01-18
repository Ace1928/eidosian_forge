import re
import codecs
import encodings
from typing import Callable, Match, Optional, Tuple, Union, cast
from w3lib._types import AnyUnicodeError, StrOrBytes
import w3lib.util
def resolve_encoding(encoding_alias: str) -> Optional[str]:
    """Return the encoding that `encoding_alias` maps to, or ``None``
    if the encoding cannot be interpreted

    >>> import w3lib.encoding
    >>> w3lib.encoding.resolve_encoding('latin1')
    'cp1252'
    >>> w3lib.encoding.resolve_encoding('gb_2312-80')
    'gb18030'
    >>>

    """
    c18n_encoding = _c18n_encoding(encoding_alias)
    translated = DEFAULT_ENCODING_TRANSLATION.get(c18n_encoding, c18n_encoding)
    try:
        return codecs.lookup(translated).name
    except LookupError:
        return None