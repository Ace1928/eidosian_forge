from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def implementer_only(*interfaces):
    """
        Create a decorator for declaring the only interfaces implemented.

        A callable is returned that makes an implements declaration on
        objects passed to it.

        .. seealso:: `zope.interface.implementer_only`
        """