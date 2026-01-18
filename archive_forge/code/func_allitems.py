import sys
def allitems(attr=None, doc=''):
    """Property for fetching attribute from all entries of container.

    Returns a property that fetches the given attributes from
    all items in a SearchIO container object.
    """

    def getter(self):
        if attr is None:
            return self._items
        return [getattr(frag, attr) for frag in self._items]
    return property(fget=getter, doc=doc)