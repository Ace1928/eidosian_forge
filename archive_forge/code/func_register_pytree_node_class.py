from __future__ import annotations
import dataclasses
import functools
import inspect
import sys
from collections import OrderedDict, defaultdict, deque, namedtuple
from operator import methodcaller
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, Sequence, overload
from typing_extensions import Self  # Python 3.11+
from optree import _C
from optree.typing import (
from optree.utils import safe_zip, total_order_sorted, unzip2
def register_pytree_node_class(cls: type[CustomTreeNode[T]] | str | None=None, *, namespace: str | None=None) -> type[CustomTreeNode[T]] | Callable[[type[CustomTreeNode[T]]], type[CustomTreeNode[T]]]:
    """Extend the set of types that are considered internal nodes in pytrees.

    See also :func:`register_pytree_node` and :func:`unregister_pytree_node`.

    The ``namespace`` argument is used to avoid collisions that occur when different libraries
    register the same Python type with different behaviors. It is recommended to add a unique prefix
    to the namespace to avoid conflicts with other libraries. Namespaces can also be used to specify
    the same class in different namespaces for different use cases.

    .. warning::
        For safety reasons, a ``namespace`` must be specified while registering a custom type. It is
        used to isolate the behavior of flattening and unflattening a pytree node type. This is to
        prevent accidental collisions between different libraries that may register the same type.

    Args:
        cls (type, optional): A Python type to treat as an internal pytree node.
        namespace (str, optional): A non-empty string that uniquely identifies the namespace of the
            type registry. This is used to isolate the registry from other modules that might
            register a different custom behavior for the same type.

    Returns:
        The same type as the input ``cls`` if the argument presents. Otherwise, return a decorator
        function that registers the class as a pytree node.

    Raises:
        TypeError: If the namespace is not a string.
        ValueError: If the namespace is an empty string.
        ValueError: If the type is already registered in the registry.

    This function is a thin wrapper around :func:`register_pytree_node`, and provides a
    class-oriented interface::

        @register_pytree_node_class(namespace='foo')
        class Special:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def tree_flatten(self):
                return ((self.x, self.y), None)

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(*children)

        @register_pytree_node_class('mylist')
        class MyList(UserList):
            def tree_flatten(self):
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                return cls(*children)
    """
    if cls is __GLOBAL_NAMESPACE or isinstance(cls, str):
        if namespace is not None:
            raise ValueError('Cannot specify `namespace` when the first argument is a string.')
        if cls == '':
            raise ValueError('The namespace cannot be an empty string.')
        return functools.partial(register_pytree_node_class, namespace=cls)
    if namespace is None:
        raise ValueError('Must specify `namespace` when the first argument is a class.')
    if namespace is not __GLOBAL_NAMESPACE and (not isinstance(namespace, str)):
        raise TypeError(f'The namespace must be a string, got {namespace}')
    if namespace == '':
        raise ValueError('The namespace cannot be an empty string.')
    if cls is None:
        return functools.partial(register_pytree_node_class, namespace=namespace)
    if not inspect.isclass(cls):
        raise TypeError(f'Expected a class, got {cls}.')
    register_pytree_node(cls, methodcaller('tree_flatten'), cls.tree_unflatten, namespace)
    return cls