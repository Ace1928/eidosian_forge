from abc import ABC, abstractmethod
from typing import List, Optional
def count_leaves(self, root):
    next_nodes = list(root.values())
    if len(next_nodes) == 0:
        return 1
    else:
        return sum([self.count_leaves(nn) for nn in next_nodes])