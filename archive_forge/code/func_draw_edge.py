from collections import Counter
from textwrap import dedent
from kombu.utils.encoding import bytes_to_str, safe_str
def draw_edge(self, a, b, scheme=None, attrs=None):
    return self.FMT(self._edge, self.label(a), self.label(b), dir=self.direction, attrs=self.attrs(attrs, self.edge_scheme))