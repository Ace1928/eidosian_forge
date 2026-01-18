from .truncatedComplex import TruncatedComplex
from snappy.snap import t3mlite as t3m
from .verificationError import *
def insert_edge_loops(self):
    position = 0
    for i in range(len(self.loop)):
        position = self.insert_edge_loop(position)
        position += 1