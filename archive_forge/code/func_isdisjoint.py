from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
def isdisjoint(self, other: 'FSM') -> bool:
    alphabet, new_to_old = self.alphabet.intersect(other.alphabet)
    initial = (self.initial, other.initial)

    def follow(current, transition):
        ss, os = current
        if ss in self.map and new_to_old[0][transition] in self.map[ss]:
            sn = self.map[ss][new_to_old[0][transition]]
        else:
            sn = None
        if os in other.map and new_to_old[1][transition] in other.map[os]:
            on = other.map[os][new_to_old[1][transition]]
        else:
            on = None
        if not sn or not on:
            raise OblivionError
        return (sn, on)

    def final(state):
        if state[0] in self.finals and state[1] in other.finals:
            raise _Marker
    try:
        crawl_hash_no_result(alphabet, initial, final, follow)
    except _Marker:
        return False
    else:
        return True