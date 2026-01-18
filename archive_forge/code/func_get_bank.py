from abc import ABC, abstractmethod
from typing import List, Optional
def get_bank(self):
    add = 0
    if self.inprogress_constraint:
        add += self.max_seqlen - self.inprogress_constraint.remaining()
    return len(self.complete_constraints) * self.max_seqlen + add