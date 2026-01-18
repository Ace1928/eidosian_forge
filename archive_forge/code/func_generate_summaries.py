import sys
import io
import copy
import array
import itertools
import struct
import zlib
from collections import namedtuple
from io import BytesIO
import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def generate_summaries(self, scale, totalSum):
    ranges = self.root.traverse()
    start, end, val = next(ranges)
    chromId = self.chromId
    chromSize = self.chromSize
    summary = _RegionSummary(chromId, start, min(start + scale, chromSize), val)
    while True:
        size = max(end - start, 1)
        totalSum.update(size, val)
        if summary.end <= start:
            summary = _RegionSummary(chromId, start, min(start + scale, chromSize), val)
        while end > summary.end:
            overlap = min(end, summary.end) - max(start, summary.start)
            assert overlap > 0
            summary.update(overlap, val)
            size -= overlap
            start = summary.end
            yield summary
            summary = _RegionSummary(chromId, start, min(start + scale, chromSize), val)
        summary.update(size, val)
        try:
            start, end, val = next(ranges)
        except StopIteration:
            break
        if summary.end <= start:
            yield summary
    yield summary