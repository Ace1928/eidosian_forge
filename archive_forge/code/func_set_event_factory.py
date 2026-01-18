def set_event_factory(trait_set, removed, added):
    """ Adapt the call signature of TraitSet.notify to create an event.

    Parameters
    ----------
    trait_set : traits.trait_set_object.TraitSet
        The set being mutated.
    removed : set
        Values removed from the set.
    added : set
        Values added to the set.

    Returns
    -------
    SetChangeEvent
    """
    return SetChangeEvent(object=trait_set, added=added, removed=removed)