from ._base import *
from .types import CreateSchemaType, UpdateSchemaType
from .models import LazyHasher, LazyUserSchema, LazyDBConfig, LazyDBSaveMetrics
@timed_cache(5)
def get_by_props(self, props: Dict[str, Any], forward=True, *args, **kwargs):
    if not props:
        return None
    search_range = range(self.idx) if forward else range(self.idx, 0, -1)
    for idx in search_range:
        if self.index.get(idx):
            item = self.index[idx]
            for name, val in props.items():
                if name not in self.schema_props:
                    continue
                if self.match_props(item, name, val, *args, **kwargs):
                    return item
    return None