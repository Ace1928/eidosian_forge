import textwrap
from pprint import PrettyPrinter
from _plotly_utils.utils import *
from _plotly_utils.data_utils import *
def node_generator(node, path=()):
    """
    General, node-yielding generator.

    Yields (node, path) tuples when it finds values that are dict
    instances.

    A path is a sequence of hashable values that can be used as either keys to
    a mapping (dict) or indices to a sequence (list). A path is always wrt to
    some object. Given an object, a path explains how to get from the top level
    of that object to a nested value in the object.

    :param (dict) node: Part of a dict to be traversed.
    :param (tuple[str]) path: Defines the path of the current node.
    :return: (Generator)

    Example:

        >>> for node, path in node_generator({'a': {'b': 5}}):
        ...     print(node, path)
        {'a': {'b': 5}} ()
        {'b': 5} ('a',)

    """
    if not isinstance(node, dict):
        return
    yield (node, path)
    for key, val in node.items():
        if isinstance(val, dict):
            for item in node_generator(val, path + (key,)):
                yield item