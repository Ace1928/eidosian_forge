from ._base import *
from .types import CreateSchemaType, UpdateSchemaType
from .models import LazyHasher, LazyUserSchema, LazyDBConfig, LazyDBSaveMetrics
def filter_items(self, item_list: List[Any], props: Dict[str, Any]=None, *args, **kwargs):
    res_list = []
    for item in item_list:
        for name, val in props.items():
            if name not in self.schema_props:
                continue
            if self.match_props(item, name, val, *args, **kwargs):
                res_list.append(item)
    return res_list