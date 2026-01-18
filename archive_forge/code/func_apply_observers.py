from traits.observation._observe import add_or_remove_notifiers
from traits.observation.expression import compile_expr
def apply_observers(object, graphs, handler, *, dispatcher, remove=False):
    """ Apply one or more ObserverGraphs to an object and handler.

    Parameters
    ----------
    object : object
        An object to be observed. Usually an instance of ``HasTraits``.
    graphs : list of ObserverGraph
        Graphs describing the observation patterns to apply.
    handler : callable(event)
        User-defined callable to handle change events.
        ``event`` is an object representing the change.
        Its type and content depends on the change.
    dispatcher : callable(callable, event).
        Callable for dispatching the user-defined handler, e.g. dispatching
        callback on a different thread.
    remove : boolean, optional
        If True, remove notifiers. i.e. unobserve the traits. The default
        is False.
    """
    for graph in graphs:
        add_or_remove_notifiers(object=object, graph=graph, handler=handler, target=object, dispatcher=dispatcher, remove=remove)