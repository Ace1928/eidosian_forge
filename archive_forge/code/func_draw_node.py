from collections import Counter
from textwrap import dedent
from kombu.utils.encoding import bytes_to_str, safe_str
def draw_node(self, obj, scheme=None, attrs=None):
    return self.FMT(self._node, self.label(obj), attrs=self.attrs(attrs, scheme))