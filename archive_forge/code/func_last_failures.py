import abc
import copy
import os
from oslo_utils import timeutils
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow import states
from taskflow.types import failure as ft
from taskflow.utils import misc
@property
def last_failures(self):
    """The last failure dictionary that was produced.

        NOTE(harlowja): This is **not** the same as the
        local ``failure`` attribute as the obtained failure dictionary in
        the ``results`` attribute (which is what this returns) is from
        associated atom failures (which is different from the directly
        related failure of the retry unit associated with this
        atom detail).
        """
    try:
        return self.results[-1][1]
    except IndexError:
        exc.raise_with_cause(exc.NotFound, 'Last failures not found')