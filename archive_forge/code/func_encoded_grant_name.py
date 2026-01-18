from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence
from lazyops.libs.dbinit.base import TextClause, text
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.exceptions import EntityExistsError, InvalidPrivilegeError
@property
def encoded_grant_name(self) -> str:
    """
        Returns the encoded name of the grantable.
        """
    return f'"{self._grant_name}"' if '-' in self._grant_name or '_' in self._grant_name else self._grant_name