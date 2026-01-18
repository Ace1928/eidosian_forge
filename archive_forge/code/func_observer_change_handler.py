from traits.constants import ComparisonMode, TraitKind
from traits.ctraits import CHasTraits
from traits.observation._observe import add_or_remove_notifiers
from traits.observation.exceptions import NotifierNotFound
from traits.trait_base import Undefined, Uninitialized
def observer_change_handler(event, graph, handler, target, dispatcher):
    """ Maintain downstream notifiers when the trait changes.

    Parameters
    ----------
    event : TraitChangeEvent
        The event that triggers this function.
    graph : ObserverGraph
        Description for the *downstream* observers, i.e. excluding the observer
        that contributed this maintainer function.
    handler : callable
        User handler.
    target : object
        Object seen by the user as the owner of the observer.
    dispatcher : callable
        Callable for dispatching the handler.
    """
    if all((event.old is not skipped for skipped in UNOBSERVABLE_VALUES)):
        try:
            add_or_remove_notifiers(object=event.old, graph=graph, handler=handler, target=target, dispatcher=dispatcher, remove=True)
        except NotifierNotFound:
            pass
    if all((event.new is not skipped for skipped in UNOBSERVABLE_VALUES)):
        add_or_remove_notifiers(object=event.new, graph=graph, handler=handler, target=target, dispatcher=dispatcher, remove=False)