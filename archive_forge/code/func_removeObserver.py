from twisted.python import log
from twisted.words.xish import xpath
def removeObserver(self, event, observerfn):
    """
        Remove callable as observer for an event.

        The observer callable is removed for all priority levels for the
        specified event.

        @param event: Event for which the observer callable was registered.
        @type event: C{str} or L{xpath.XPathQuery}
        @param observerfn: Observer callable to be unregistered.
        """
    if self._dispatchDepth > 0:
        self._updateQueue.append(lambda: self.removeObserver(event, observerfn))
        return
    event, observers = self._getEventAndObservers(event)
    emptyLists = []
    for priority, priorityObservers in observers.items():
        for query, callbacklist in priorityObservers.items():
            if event == query:
                callbacklist.removeCallback(observerfn)
                if callbacklist.isEmpty():
                    emptyLists.append((priority, query))
    for priority, query in emptyLists:
        del observers[priority][query]