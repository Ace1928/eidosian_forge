from typing import Iterable, List, Tuple, Union, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from constantly import NamedConstant
from twisted.trial import unittest
from .._filter import (
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._observer import LogPublisher, bitbucketLogObserver
def filterWith(self, filters: Iterable[str], other: bool=False) -> Union[List[int], Tuple[List[int], List[int]]]:
    """
        Apply a set of pre-defined filters on a known set of events and return
        the filtered list of event numbers.

        The pre-defined events are four events with a C{count} attribute set to
        C{0}, C{1}, C{2}, and C{3}.

        @param filters: names of the filters to apply.
            Options are:
                - C{"twoMinus"} (count <=2),
                - C{"twoPlus"} (count >= 2),
                - C{"notTwo"} (count != 2),
                - C{"no"} (False).
        @param other: Whether to return a list of filtered events as well.

        @return: event numbers or 2-tuple of lists of event numbers.
        """
    events: List[LogEvent] = [dict(count=0), dict(count=1), dict(count=2), dict(count=3)]

    class Filters:

        @staticmethod
        def twoMinus(event: LogEvent) -> NamedConstant:
            """
                count <= 2

                @param event: an event

                @return: L{PredicateResult.yes} if C{event["count"] <= 2},
                    otherwise L{PredicateResult.maybe}.
                """
            if event['count'] <= 2:
                return PredicateResult.yes
            return PredicateResult.maybe

        @staticmethod
        def twoPlus(event: LogEvent) -> NamedConstant:
            """
                count >= 2

                @param event: an event

                @return: L{PredicateResult.yes} if C{event["count"] >= 2},
                    otherwise L{PredicateResult.maybe}.
                """
            if event['count'] >= 2:
                return PredicateResult.yes
            return PredicateResult.maybe

        @staticmethod
        def notTwo(event: LogEvent) -> NamedConstant:
            """
                count != 2

                @param event: an event

                @return: L{PredicateResult.yes} if C{event["count"] != 2},
                    otherwise L{PredicateResult.maybe}.
                """
            if event['count'] == 2:
                return PredicateResult.no
            return PredicateResult.maybe

        @staticmethod
        def no(event: LogEvent) -> NamedConstant:
            """
                No way, man.

                @param event: an event

                @return: L{PredicateResult.no}
                """
            return PredicateResult.no

        @staticmethod
        def bogus(event: LogEvent) -> NamedConstant:
            """
                Bogus result.

                @param event: an event

                @return: something other than a valid predicate result.
                """
            return None
    predicates = (getattr(Filters, f) for f in filters)
    eventsSeen: List[LogEvent] = []
    eventsNotSeen: List[LogEvent] = []
    trackingObserver = cast(ILogObserver, eventsSeen.append)
    if other:
        negativeObserver = cast(ILogObserver, eventsNotSeen.append)
    else:
        negativeObserver = bitbucketLogObserver
    filteringObserver = FilteringLogObserver(trackingObserver, predicates, negativeObserver)
    for e in events:
        filteringObserver(e)
    if other:
        return ([cast(int, e['count']) for e in eventsSeen], [cast(int, e['count']) for e in eventsNotSeen])
    else:
        return [cast(int, e['count']) for e in eventsSeen]