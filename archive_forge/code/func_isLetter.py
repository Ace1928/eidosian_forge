from ..common.html_re import HTML_TAG_RE
from ..common.utils import isLinkClose, isLinkOpen
from .state_inline import StateInline
def isLetter(ch: int) -> bool:
    lc = ch | 32
    return lc >= 97 and lc <= 122