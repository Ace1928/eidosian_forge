from collections import deque
from typing import List, Set
def forward_transitive_closure(self, src: str) -> Set[str]:
    """Returns a set of nodes that are reachable from src"""
    result = set(src)
    working_set = deque(src)
    while len(working_set) > 0:
        cur = working_set.popleft()
        for n in self.successors(cur):
            if n not in result:
                result.add(n)
                working_set.append(n)
    return result