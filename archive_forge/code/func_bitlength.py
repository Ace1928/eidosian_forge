import re
import itertools
@property
def bitlength(self):
    if self.bits is None:
        return 0
    q, r = divmod(len(self), len(self.bits))
    return q * sum(self.bits) + sum(self.bits[:r])