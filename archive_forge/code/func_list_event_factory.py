def list_event_factory(trait_list, index, removed, added):
    """ Adapt the call signature of TraitList.notify to create an event.

    Parameters
    ----------
    trait_list : traits.trait_list_object.TraitList
        TraitList object being mutated.
    index : int or slice
        The indices being modified by the operation.
    removed : list
        The items removed from the list.
    added : list
        The items added to the list.

    Returns
    -------
    ListChangeEvent
    """
    return ListChangeEvent(object=trait_list, index=index, removed=removed, added=added)