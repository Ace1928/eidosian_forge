from .truncatedComplex import TruncatedComplex
from snappy.snap import t3mlite as t3m
from .verificationError import *
def compute_loop(self):
    self.expand()
    self.shift_loop_to_start_with_edge_loop()
    self.truncated_complex.check_loop(self.loop)
    return self.loop