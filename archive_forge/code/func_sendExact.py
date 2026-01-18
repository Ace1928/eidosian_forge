import weakref
from pydispatch import saferef, robustapply, errors
def sendExact(signal=Any, sender=Anonymous, *arguments, **named):
    """Send signal only to those receivers registered for exact message

    sendExact allows for avoiding Any/Anonymous registered
    handlers, sending only to those receivers explicitly
    registered for a particular signal on a particular
    sender.
    """
    responses = []
    for receiver in liveReceivers(getReceivers(sender, signal)):
        response = robustapply.robustApply(receiver, *arguments, signal=signal, sender=sender, **named)
        responses.append((receiver, response))
    return responses