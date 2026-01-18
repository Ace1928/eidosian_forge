import snappy
import sys
import spherogram.graphs as graphs
from spherogram import DTcodec
def test_Marc():
    for dt in dtcodes:
        L = DTcodec(dt)