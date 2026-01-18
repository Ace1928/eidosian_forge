import os
from typing import (
from blib2to3.pgen2 import grammar, token, tokenize
from blib2to3.pgen2.tokenize import GoodTokenInfo
def addarc(self, next: 'DFAState', label: str) -> None:
    assert isinstance(label, str)
    assert label not in self.arcs
    assert isinstance(next, DFAState)
    self.arcs[label] = next