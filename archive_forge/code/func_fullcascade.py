import sys
def fullcascade(attr, doc=''):
    """Return a getter property with a cascading setter.

    This is similar to ``optionalcascade``, but for SearchIO containers that have
    at least one item (HSP). The getter always retrieves the attribute
    value from the first item. If the items have more than one attribute values,
    an error will be raised. The setter behaves like ``partialcascade``, except
    that it only sets attributes to items in the object, not the object itself.

    """

    def getter(self):
        return getattr(self._items[0], attr)

    def setter(self, value):
        for item in self:
            setattr(item, attr, value)
    return property(fget=getter, fset=setter, doc=doc)