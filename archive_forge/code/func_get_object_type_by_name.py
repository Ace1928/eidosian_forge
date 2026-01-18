from abc import ABC, abstractmethod
import calendar
from collections import deque
from datetime import datetime, timedelta, tzinfo
from string import digits
import re
import time
import warnings
from git.util import IterableList, IterableObj, Actor
from typing import (
from git.types import Has_id_attribute, Literal  # , _T
def get_object_type_by_name(object_type_name: bytes) -> Union[Type['Commit'], Type['TagObject'], Type['Tree'], Type['Blob']]:
    """
    :return: A type suitable to handle the given object type name.
        Use the type to create new instances.

    :param object_type_name: Member of TYPES

    :raise ValueError: If object_type_name is unknown
    """
    if object_type_name == b'commit':
        from . import commit
        return commit.Commit
    elif object_type_name == b'tag':
        from . import tag
        return tag.TagObject
    elif object_type_name == b'blob':
        from . import blob
        return blob.Blob
    elif object_type_name == b'tree':
        from . import tree
        return tree.Tree
    else:
        raise ValueError('Cannot handle unknown object type: %s' % object_type_name.decode())