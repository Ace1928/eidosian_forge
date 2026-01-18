from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def registered(required, provided, name=''):
    """Return the component registered for the given interfaces and name

        name must be text.

        Unlike the lookup method, this methods won't retrieve
        components registered for more specific required interfaces or
        less specific provided interfaces.

        If no component was registered exactly for the given
        interfaces and name, then None is returned.

        """