from __future__ import annotations
from itertools import islice
from operator import itemgetter
from threading import Lock
from typing import Any
def sort_heap(self, h: list[tuple[int, str]]) -> tuple[int, str]:
    """Sort heap of events.

        List of tuples containing at least two elements, representing
        an event, where the first element is the event's scalar clock value,
        and the second element is the id of the process (usually
        ``"hostname:pid"``): ``sh([(clock, processid, ...?), (...)])``

        The list must already be sorted, which is why we refer to it as a
        heap.

        The tuple will not be unpacked, so more than two elements can be
        present.

        Will return the latest event.
        """
    if h[0][0] == h[1][0]:
        same = []
        for PN in zip(h, islice(h, 1, None)):
            if PN[0][0] != PN[1][0]:
                break
            same.append(PN[0])
        return sorted(same, key=lambda event: event[1])[0]
    return h[0]