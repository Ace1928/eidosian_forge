from functools import partial
from typing import Dict, Iterable
from zope.interface import Interface, implementer
from constantly import NamedConstant, Names
from ._interfaces import ILogObserver, LogEvent
from ._levels import InvalidLogLevelError, LogLevel
from ._observer import bitbucketLogObserver
def shouldLogEvent(predicates: Iterable[ILogFilterPredicate], event: LogEvent) -> bool:
    """
    Determine whether an event should be logged, based on the result of
    C{predicates}.

    By default, the result is C{True}; so if there are no predicates,
    everything will be logged.

    If any predicate returns C{yes}, then we will immediately return C{True}.

    If any predicate returns C{no}, then we will immediately return C{False}.

    As predicates return C{maybe}, we keep calling the next predicate until we
    run out, at which point we return C{True}.

    @param predicates: The predicates to use.
    @param event: An event

    @return: True if the message should be forwarded on, C{False} if not.
    """
    for predicate in predicates:
        result = predicate(event)
        if result == PredicateResult.yes:
            return True
        if result == PredicateResult.no:
            return False
        if result == PredicateResult.maybe:
            continue
        raise TypeError(f'Invalid predicate result: {result!r}')
    return True