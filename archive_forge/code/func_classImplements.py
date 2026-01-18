from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def classImplements(class_, *interfaces):
    """
        Declare additional interfaces implemented for instances of a class.

        The arguments after the class are one or more interfaces or
        interface specifications (`IDeclaration` objects).

        The interfaces given (including the interfaces in the
        specifications) are added to any interfaces previously
        declared.

        Consider the following example::

          class C(A, B):
             ...

          classImplements(C, I1, I2)


        Instances of ``C`` provide ``I1``, ``I2``, and whatever interfaces
        instances of ``A`` and ``B`` provide. This is equivalent to::

            @implementer(I1, I2)
            class C(A, B):
                pass

        .. seealso:: `zope.interface.classImplements`
        .. seealso:: `zope.interface.implementer`
        """