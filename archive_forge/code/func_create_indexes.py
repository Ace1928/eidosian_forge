from ._base import *
from .models import LazyDBCacheBase
def create_indexes(self, dbdata):
    for n, i in enumerate(dbdata):
        if i not in self.all_wks:
            idx = dbdata[i]
            wks = self.sheet.add_worksheet(i, index=n)
            wks.append_row(idx.schema_props)