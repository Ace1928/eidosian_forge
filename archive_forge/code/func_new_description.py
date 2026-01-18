from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
@property
def new_description(self) -> TopologyDescription:
    """The new
        :class:`~pymongo.topology_description.TopologyDescription`.
        """
    return self.__new_description