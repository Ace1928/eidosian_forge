from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping, Optional
from bson._helpers import _getstate_slots, _setstate_slots
from bson.son import SON
def as_doc(self) -> SON[str, Any]:
    """Get the SON document representation of this DBRef.

        Generally not needed by application developers
        """
    doc = SON([('$ref', self.collection), ('$id', self.id)])
    if self.database is not None:
        doc['$db'] = self.database
    doc.update(self.__kwargs)
    return doc