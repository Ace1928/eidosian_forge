from typing import (
from blib2to3.pgen2.grammar import Grammar
import sys
from io import StringIO
def generate_matches(self, nodes: List[NL]) -> Iterator[Tuple[int, _Results]]:
    if self.content is None:
        if len(nodes) == 0:
            yield (0, {})
    else:
        for c, r in self.content.generate_matches(nodes):
            return
        yield (0, {})