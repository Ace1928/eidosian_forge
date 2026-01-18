from ..core import BaseRenderer
from ..util import escape as escape_text, striptags, safe_entity
def blank_line(self) -> str:
    return ''