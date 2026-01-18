import collections
import functools
import operator
from ovs.db import data
def index_entry_from_row(self, row):
    return row._table.rows.IndexEntry(uuid=row.uuid, **{c.column: getattr(row, c.column) for c in self.columns})