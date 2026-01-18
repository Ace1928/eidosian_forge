from typing import NamedTuple
from typing import Sequence, Tuple
@property
def num_copies(self):
    """The total number of copies of any shot quantity."""
    return sum((s.copies for s in self.shot_vector))