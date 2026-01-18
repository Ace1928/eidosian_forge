import weakref
from pydispatch import saferef, robustapply, errors
def getAllReceivers(sender=Any, signal=Any):
    """Get list of all receivers from global tables

    This gets all receivers which should receive
    the given signal from sender, each receiver should
    be produced only once by the resulting generator
    """
    receivers = {}
    for set in (getReceivers(sender, signal), getReceivers(sender, Any), getReceivers(Any, signal), getReceivers(Any, Any)):
        for receiver in set:
            if receiver:
                try:
                    if receiver not in receivers:
                        receivers[receiver] = 1
                        yield receiver
                except TypeError:
                    pass