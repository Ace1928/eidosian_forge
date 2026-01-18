from __future__ import annotations
import time
from ..types.properties import StatefulProperty
from ..types.user_session import UserSession
from ..utils.lazy import logger
from .admin import AZManagementClient
from typing import List, Optional, Any, Dict, TYPE_CHECKING
@property
def pdict_s(self) -> 'PersistentDict':
    """
        Returns the Persistent Dict for Session Mapping
        """
    if self._pdict_s is None:
        self._pdict_s = self.pdict.get_child('mapping')
    return self._pdict_s