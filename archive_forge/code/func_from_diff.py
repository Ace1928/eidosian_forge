from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
@classmethod
def from_diff(cls, src, dst, optimization=True, dumps=None, pointer_cls=JsonPointer):
    """Creates JsonPatch instance based on comparison of two document
        objects. Json patch would be created for `src` argument against `dst`
        one.

        :param src: Data source document object.
        :type src: dict

        :param dst: Data source document object.
        :type dst: dict

        :param dumps: A function of one argument that produces a serialized
                      JSON string.
        :type dumps: function

        :param pointer_cls: JSON pointer class to use.
        :type pointer_cls: Type[JsonPointer]

        :return: :class:`JsonPatch` instance.

        >>> src = {'foo': 'bar', 'numbers': [1, 3, 4, 8]}
        >>> dst = {'baz': 'qux', 'numbers': [1, 4, 7]}
        >>> patch = JsonPatch.from_diff(src, dst)
        >>> new = patch.apply(src)
        >>> new == dst
        True
        """
    json_dumper = dumps or cls.json_dumper
    builder = DiffBuilder(src, dst, json_dumper, pointer_cls=pointer_cls)
    builder._compare_values('', None, src, dst)
    ops = list(builder.execute())
    return cls(ops, pointer_cls=pointer_cls)