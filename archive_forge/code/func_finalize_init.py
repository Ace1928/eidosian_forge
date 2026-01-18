from ._base import *
from .types import CreateSchemaType, UpdateSchemaType
from .models import LazyHasher, LazyUserSchema, LazyDBConfig, LazyDBSaveMetrics
def finalize_init(self):
    self.db = LazyObject(self._db)
    for name, index in self._db.items():
        schema_name = f'{name}_schema'
        setattr(self, schema_name, self._base_schemas[name])
        for method_type in ['get', 'create', 'remove', 'update']:
            method_name = f'{name}_{method_type}'
            method_func = lambda method_index=name, method_type=method_type, *args, **kwargs: self._db[method_index](*args, method=method_type, **kwargs)
            setattr(self, method_name, method_func)
    self.env = LazyEnv
    if self.config.autosave:
        if self.env.is_threadsafe:
            self.start_background()
        else:
            logger.warn(f'[{self.dbname} Finalize]: Currently not threadsafe. Call .start_background in main thread.')