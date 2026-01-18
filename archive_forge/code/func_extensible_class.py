from typing import Callable, Dict, Optional, Type, get_type_hints, Any
import inspect
from triad.utils.assertion import assert_or_throw
def extensible_class(class_type: Type) -> Type:
    """The decorator making classes extensible by external methods

    :param class_type: the class under the decorator
    :return: the ``class_type``

    .. admonition:: Examples

        .. code-block:: python

            @extensible_class
            class A:

                # It's recommended to implement __getattr__ so that
                # PyLint will not complain about the dynamically added methods
                def __getattr__(self, name):
                    raise NotImplementedError

            @extension_method
            def method(obj:A):
                return 1

            assert 1 == A().method()

    .. note::

        If the method name is already in the original class, a ValueError will be
        thrown. You can't modify any built-in attribute.
    """
    _CLASS_EXTENSIONS.register_type(class_type)
    return class_type