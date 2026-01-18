import inspect
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, overload
from scrapy.exceptions import ScrapyDeprecationWarning
def method_is_overridden(subclass: type, base_class: type, method_name: str) -> bool:
    """
    Return True if a method named ``method_name`` of a ``base_class``
    is overridden in a ``subclass``.

    >>> class Base:
    ...     def foo(self):
    ...         pass
    >>> class Sub1(Base):
    ...     pass
    >>> class Sub2(Base):
    ...     def foo(self):
    ...         pass
    >>> class Sub3(Sub1):
    ...     def foo(self):
    ...         pass
    >>> class Sub4(Sub2):
    ...     pass
    >>> method_is_overridden(Sub1, Base, 'foo')
    False
    >>> method_is_overridden(Sub2, Base, 'foo')
    True
    >>> method_is_overridden(Sub3, Base, 'foo')
    True
    >>> method_is_overridden(Sub4, Base, 'foo')
    True
    """
    base_method = getattr(base_class, method_name)
    sub_method = getattr(subclass, method_name)
    return base_method.__code__ is not sub_method.__code__