def add_or_remove_notifiers(*, object, graph, handler, target, dispatcher, remove):
    """ Add/Remove notifiers on objects following the description on an
    ObserverGraph.

    All nodes in ``ObserverGraph`` are required to be instances of
    ``IObserver``. The interface of ``IObserver`` supports this function.

    Parameters
    ----------
    object : object
        An object to be observed.
    graph : ObserverGraph
        A graph describing what and how extended traits are being observed.
        All nodes must be ``IObserver``.
    handler : callable(event)
        User-defined callable to handle change events.
        ``event`` is an object representing the change.
        Its type and content depends on the change.
    target : Any
        An object for defining the context of the user's handler notifier.
        This is typically an instance of HasTraits seen by the user as the
        "owner" of the observer.
    dispatcher : callable(callable, event)
        Callable for dispatching the user-defined handler, e.g. dispatching
        callback on a different thread.
    remove : boolean
        If true, notifiers are being removed.

    Raises
    ------
    NotiferNotFound
        Raised when notifier cannot be found for removal.
    """
    callable_ = _AddOrRemoveNotifier(object=object, graph=graph, handler=handler, target=target, dispatcher=dispatcher, remove=remove)
    callable_()