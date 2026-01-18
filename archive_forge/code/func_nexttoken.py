import logging
import re
from typing import (
from . import settings
from .utils import choplist
def nexttoken(self) -> Tuple[int, PSBaseParserToken]:
    while not self._tokens:
        self.fillbuf()
        self.charpos = self._parse1(self.buf, self.charpos)
    token = self._tokens.pop(0)
    log.debug('nexttoken: %r', token)
    return token