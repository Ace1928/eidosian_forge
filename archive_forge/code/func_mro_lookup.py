from functools import reduce
def mro_lookup(cls, attr, stop=None, monkey_patched=None):
    """Return the first node by MRO order that defines an attribute.

    Arguments:
        cls (Any): Child class to traverse.
        attr (str): Name of attribute to find.
        stop (Set[Any]): A set of types that if reached will stop
            the search.
        monkey_patched (Sequence): Use one of the stop classes
            if the attributes module origin isn't in this list.
            Used to detect monkey patched attributes.

    Returns:
        Any: The attribute value, or :const:`None` if not found.
    """
    stop = set() if not stop else stop
    monkey_patched = [] if not monkey_patched else monkey_patched
    for node in cls.mro():
        if node in stop:
            try:
                value = node.__dict__[attr]
                module_origin = value.__module__
            except (AttributeError, KeyError):
                pass
            else:
                if module_origin not in monkey_patched:
                    return node
            return
        if attr in node.__dict__:
            return node