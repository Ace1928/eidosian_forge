from ._base import *
from .types import CreateSchemaType, UpdateSchemaType
from .models import LazyHasher, LazyUserSchema, LazyDBConfig, LazyDBSaveMetrics
@property
def dbname(self):
    return self.__class__.__name__