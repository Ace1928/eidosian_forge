from __future__ import annotations
import typing as ty
from nibabel.deprecated import deprecate_with_version
def auto_attr(func: ty.Callable[[InstanceT], T]) -> OneTimeProperty[T]:
    """Decorator to create OneTimeProperty attributes.

    Parameters
    ----------
      func : method
        The method that will be called the first time to compute a value.
        Afterwards, the method's name will be a standard attribute holding the
        value of this computation.

    Examples
    --------
    >>> class MagicProp:
    ...     @auto_attr
    ...     def a(self):
    ...         return 99
    ...
    >>> x = MagicProp()
    >>> 'a' in x.__dict__
    False
    >>> x.a
    99
    >>> 'a' in x.__dict__
    True
    """
    return OneTimeProperty(func)