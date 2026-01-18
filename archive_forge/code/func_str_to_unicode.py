from warnings import warn
from typing import Optional
from w3lib._types import StrOrBytes
def str_to_unicode(text: StrOrBytes, encoding: Optional[str]=None, errors: str='strict') -> str:
    warn('The w3lib.utils.str_to_unicode function is deprecated and will be removed in a future release.', DeprecationWarning, stacklevel=2)
    if encoding is None:
        encoding = 'utf-8'
    if isinstance(text, bytes):
        return text.decode(encoding, errors)
    return text