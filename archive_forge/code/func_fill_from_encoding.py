from typing import Dict, List
from .adobe_glyphs import adobe_glyphs
from .pdfdoc import _pdfdoc_encoding
from .std import _std_encoding
from .symbol import _symbol_encoding
from .zapfding import _zapfding_encoding
def fill_from_encoding(enc: str) -> List[str]:
    lst: List[str] = []
    for x in range(256):
        try:
            lst += (bytes((x,)).decode(enc),)
        except Exception:
            lst += (chr(x),)
    return lst