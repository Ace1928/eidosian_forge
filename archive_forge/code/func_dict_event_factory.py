def dict_event_factory(trait_dict, removed, added, changed):
    """ Adapt the call signature of TraitDict.notify to create an event.

    Parameters
    ----------
    trait_dict : traits.trait_dict_object.TraitDict
        The dict being mutated.
    removed : dict
        Items removed from the dict
    added : dict
        Items added to the dict
    changed : dict
        Old values for items updated on the dict.

    Returns
    -------
    DictChangeEvent
    """
    removed = removed.copy()
    removed.update(changed)
    for key in changed:
        added[key] = trait_dict[key]
    return DictChangeEvent(object=trait_dict, added=added, removed=removed)