from snappy import *
from snappy.SnapPy import triangulate_link_complement_from_data
from spherogram import FatGraph, FatEdge, CyclicList, Link, Crossing
import string
def first_under(self):
    first, second, even_over = self
    if even_over:
        return first - 1 if first % 2 == 1 else second - 1
    else:
        return first - 1 if first % 2 == 0 else second - 1