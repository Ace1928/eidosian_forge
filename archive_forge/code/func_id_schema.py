from ._base import *
from .types import CreateSchemaType, UpdateSchemaType
from .models import LazyHasher, LazyUserSchema, LazyDBConfig, LazyDBSaveMetrics
@property
def id_schema(self):
    return {'uid': (int, Field(default_factory=self.get_idx)), 'dbid': (str, Field(default_factory=self.get_dbid)), 'created': (str, Field(default_factory=self.get_timestamp)), 'updated': (str, Field(default_factory=self.get_timestamp))}