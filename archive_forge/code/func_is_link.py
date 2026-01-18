from __future__ import absolute_import, print_function, unicode_literals
import typing
from typing import cast
import six
from copy import deepcopy
from ._typing import Text, overload
from .enums import ResourceType
from .errors import MissingInfoNamespace
from .path import join
from .permissions import Permissions
from .time import epoch_to_datetime
@property
def is_link(self):
    """`bool`: `True` if the resource is a symlink."""
    self._require_namespace('link')
    return self.get('link', 'target', None) is not None