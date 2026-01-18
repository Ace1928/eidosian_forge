from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def directlyProvides(object, *interfaces):
    """
        Declare interfaces declared directly for an object.

        The arguments after the object are one or more interfaces or
        interface specifications (`IDeclaration` objects).

        .. caution::
           The interfaces given (including the interfaces in the
           specifications) *replace* interfaces previously
           declared for the object. See :meth:`alsoProvides` to add
           additional interfaces.

        Consider the following example::

          class C(A, B):
             ...

          ob = C()
          directlyProvides(ob, I1, I2)

        The object, ``ob`` provides ``I1``, ``I2``, and whatever interfaces
        instances have been declared for instances of ``C``.

        To remove directly provided interfaces, use `directlyProvidedBy` and
        subtract the unwanted interfaces. For example::

          directlyProvides(ob, directlyProvidedBy(ob)-I2)

        removes I2 from the interfaces directly provided by
        ``ob``. The object, ``ob`` no longer directly provides ``I2``,
        although it might still provide ``I2`` if it's class
        implements ``I2``.

        To add directly provided interfaces, use `directlyProvidedBy` and
        include additional interfaces.  For example::

          directlyProvides(ob, directlyProvidedBy(ob), I2)

        adds I2 to the interfaces directly provided by ob.

        .. seealso:: `zope.interface.directlyProvides`
        """