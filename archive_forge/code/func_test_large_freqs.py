from __future__ import division
import pytest
from preshed.counter import PreshCounter
import os
def test_large_freqs():
    if 'TEST_FILE_LOC' in os.environ:
        loc = os.environ['TEST_FILE_LOC']
    else:
        return None
    counts = PreshCounter()
    for i, line in enumerate(open(loc)):
        line = line.strip()
        if not line:
            continue
        freq = int(line.split()[0])
        counts.inc(i + 1, freq)
    oov = i + 2
    assert counts.prob(oov) == 0.0
    assert counts.prob(1) < 0.1
    counts.smooth()
    assert counts.prob(oov) > 0
    assert counts.prob(oov) < counts.prob(i)