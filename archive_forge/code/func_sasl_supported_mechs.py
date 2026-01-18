from __future__ import annotations
import copy
import datetime
import itertools
from typing import Any, Generic, Mapping, Optional
from bson.objectid import ObjectId
from pymongo import common
from pymongo.server_type import SERVER_TYPE
from pymongo.typings import ClusterTime, _DocumentType
@property
def sasl_supported_mechs(self) -> list[str]:
    """Supported authentication mechanisms for the current user.

        For example::

            >>> hello.sasl_supported_mechs
            ["SCRAM-SHA-1", "SCRAM-SHA-256"]

        """
    return self._doc.get('saslSupportedMechs', [])