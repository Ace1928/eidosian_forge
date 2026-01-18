from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
@property
def tag_sets(self) -> _TagSets:
    """Set ``tag_sets`` to a list of dictionaries like [{'dc': 'ny'}] to
        read only from members whose ``dc`` tag has the value ``"ny"``.
        To specify a priority-order for tag sets, provide a list of
        tag sets: ``[{'dc': 'ny'}, {'dc': 'la'}, {}]``. A final, empty tag
        set, ``{}``, means "read from any member that matches the mode,
        ignoring tags." MongoClient tries each set of tags in turn
        until it finds a set of tags with at least one matching member.
        For example, to only send a query to an analytic node::

           Nearest(tag_sets=[{"node":"analytics"}])

        Or using :class:`SecondaryPreferred`::

           SecondaryPreferred(tag_sets=[{"node":"analytics"}])

           .. seealso:: `Data-Center Awareness
               <https://www.mongodb.com/docs/manual/data-center-awareness/>`_
        """
    return list(self.__tag_sets) if self.__tag_sets else [{}]