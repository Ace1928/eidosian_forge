from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
def connect_all(i, substate):
    """
                Take a state in the numbered FSM and return a set containing it, plus
                (if it's final) the first state from the next FSM, plus (if that's
                final) the first state from the next but one FSM, plus...
            """
    result = {(i, substate)}
    while i < last_index and substate in fsms[i].finals:
        i += 1
        substate = fsms[i].initial
        result.add((i, substate))
    return result