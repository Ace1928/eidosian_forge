from ._base import *
from .types import CreateSchemaType, UpdateSchemaType
from .models import LazyHasher, LazyUserSchema, LazyDBConfig, LazyDBSaveMetrics
@classmethod
def create_hash(cls, prop=None, hash_prop=None, item=None, item_data=None, val=None, *args, **kw):
    if item is None and item_data is None and (val is None):
        raise ValueError
    if val:
        return LazyHasher.create(val)
    item_data = item_data or jsonable_encoder(item)
    hash_results = LazyHasher.create(item_data[prop])
    if not item:
        return {hash_prop: hash_results}
    setattr(item, hash_prop, LazyHasher.create[item_data[prop]])
    return item