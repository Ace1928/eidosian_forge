import os
from typing import (
from blib2to3.pgen2 import grammar, token, tokenize
from blib2to3.pgen2.tokenize import GoodTokenInfo
def gettoken(self) -> None:
    tup = next(self.generator)
    while tup[0] in (tokenize.COMMENT, tokenize.NL):
        tup = next(self.generator)
    self.type, self.value, self.begin, self.end, self.line = tup