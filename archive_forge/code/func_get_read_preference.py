from __future__ import annotations
from collections.abc import Callable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Optional, Union
from bson.son import SON
from pymongo import common
from pymongo.collation import validate_collation_or_none
from pymongo.errors import ConfigurationError
from pymongo.read_preferences import ReadPreference, _AggWritePref
def get_read_preference(self, session: Optional[ClientSession]) -> Union[_AggWritePref, _ServerMode]:
    if self._write_preference:
        return self._write_preference
    pref = self._target._read_preference_for(session)
    if self._performs_write and pref != ReadPreference.PRIMARY:
        self._write_preference = pref = _AggWritePref(pref)
    return pref