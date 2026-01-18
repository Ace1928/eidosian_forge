from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def del_column(self, fieldname) -> None:
    """Delete a column from the table

        Arguments:

        fieldname - The field name of the column you want to delete."""
    if fieldname not in self._field_names:
        msg = "Can't delete column {!r} which is not a field name of this table. Field names are: {}".format(fieldname, ', '.join(map(repr, self._field_names)))
        raise ValueError(msg)
    col_index = self._field_names.index(fieldname)
    del self._field_names[col_index]
    for row in self._rows:
        del row[col_index]