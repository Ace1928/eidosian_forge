from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def directlyProvidedBy(object):
    """
        Return the interfaces directly provided by the given object.

        The value returned is an `IDeclaration`.

        .. seealso:: `zope.interface.directlyProvidedBy`
        """