from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def noLongerProvides(object, interface):
    """
        Remove an interface from the list of an object's directly provided
        interfaces.

        For example::

          noLongerProvides(ob, I1)

        is equivalent to::

          directlyProvides(ob, directlyProvidedBy(ob) - I1)

        with the exception that if ``I1`` is an interface that is
        provided by ``ob`` through the class's implementation,
        `ValueError` is raised.

        .. seealso:: `zope.interface.noLongerProvides`
        """