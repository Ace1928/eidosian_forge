from .links_base import Strand, Crossing, Link
import random
import collections
def insert_space(self, i, n):
    """
        Shift indices upwards when necessary so that the n slots

        i, i + 1, ... , i + n - 1

        are unassigned.
        """
    assert isinstance(i, int)

    def shift(j):
        return j if j < i else j + n
    self.n = self.n + n
    self.int_to_set = {shift(k): v for k, v in self.int_to_set.items()}
    self.set_to_int = {k: shift(v) for k, v in self.set_to_int.items()}