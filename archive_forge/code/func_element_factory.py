from decimal import Decimal
from boto.compat import filter, map
def element_factory(self, name, parent):

    class DynamicElement(parent):
        _name = name
    setattr(DynamicElement, '__name__', str(name))
    return DynamicElement