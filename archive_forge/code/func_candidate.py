import itertools
from functools import update_wrapper
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from .entry_points import load_entry_point
def candidate(self, matcher: Callable[..., bool], priority: float=1.0) -> Callable:
    """A decorator to register a child function with matcher and priority.

        .. note::

            The order to be matched is determined by both the priority
            and the order of registration.

            * The default priority is 1.0
            * Children with higher priority values will be matched earlier
            * When ``priority>0`` then later registrations will be matched earlier
            * When ``priority<=0`` then earlier registrations will be matched earlier

            So if you want to 'overwrite' the existed matches, set priority to be
            greater than 1.0. If you want to 'ignore' the current if there are other
            matches, set priority to 0.0.

        .. seealso::

            Please see examples in :func:`.conditional_dispatcher` and
            :func:`.conditional_broadcaster`.

        :param matcher: a function determines whether it is a match
            based on the same input as the parent function
        :param priority: it determines the order to be matched,
            **higher value means higher priority**, defaults to 1.0
        """

    def _run(_func: Callable[..., Any]):
        self.register(_func, matcher=matcher, priority=priority)
        return _func
    return _run