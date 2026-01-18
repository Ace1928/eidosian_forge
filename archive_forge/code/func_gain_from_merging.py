from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def gain_from_merging(self, other_encoding):
    combined_chars = other_encoding.chars | self.chars
    combined_width = bit_count(combined_chars)
    combined_columns = self.columns | other_encoding.columns
    combined_overhead = _Encoding._characteristic_overhead(combined_columns)
    combined_gain = +self.overhead + other_encoding.overhead - combined_overhead - (combined_width - self.width) * len(self.items) - (combined_width - other_encoding.width) * len(other_encoding.items)
    return combined_gain