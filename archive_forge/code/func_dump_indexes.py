from ._base import *
from .models import LazyDBCacheBase
def dump_indexes(self, dbdata):
    for schema, idx in dbdata.items():
        wks = self.sheet.worksheet(schema)
        items = []
        for i in idx.index.values():
            d = i.dict()
            items.append(list(d.values()))
        wks.insert_rows(items, row=2)
    self.refresh_index()