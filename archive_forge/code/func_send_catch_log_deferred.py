import collections.abc
import logging
from typing import Any as TypingAny
from typing import List, Tuple
from pydispatch.dispatcher import (
from pydispatch.robustapply import robustApply
from twisted.internet.defer import Deferred, DeferredList
from twisted.python.failure import Failure
from scrapy.exceptions import StopDownload
from scrapy.utils.defer import maybeDeferred_coro
from scrapy.utils.log import failure_to_exc_info
def send_catch_log_deferred(signal: TypingAny=Any, sender: TypingAny=Anonymous, *arguments: TypingAny, **named: TypingAny) -> Deferred:
    """Like send_catch_log but supports returning deferreds on signal handlers.
    Returns a deferred that gets fired once all signal handlers deferreds were
    fired.
    """

    def logerror(failure: Failure, recv: Any) -> Failure:
        if dont_log is None or not isinstance(failure.value, dont_log):
            logger.error('Error caught on signal handler: %(receiver)s', {'receiver': recv}, exc_info=failure_to_exc_info(failure), extra={'spider': spider})
        return failure
    dont_log = named.pop('dont_log', None)
    spider = named.get('spider', None)
    dfds = []
    for receiver in liveReceivers(getAllReceivers(sender, signal)):
        d = maybeDeferred_coro(robustApply, receiver, *arguments, signal=signal, sender=sender, **named)
        d.addErrback(logerror, receiver)
        d.addBoth(lambda result: (receiver, result))
        dfds.append(d)
    d = DeferredList(dfds)
    d.addCallback(lambda out: [x[1] for x in out])
    return d