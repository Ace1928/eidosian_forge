from ._base import *
from .types import CreateSchemaType, UpdateSchemaType
from .models import LazyHasher, LazyUserSchema, LazyDBConfig, LazyDBSaveMetrics
def setup_db_schema(self):
    self._db = {}
    self._base_schemas = {}
    self.__class__.__name__ = 'LazyDB' if not self.config.dbname else f'{self.config.dbname}_LazyDB'
    logger.info(f'Initializing {self.__class__.__name__}')
    logger.info(f'[{self.dbname} Setup]: Setting Up DB Schema')
    if self.config.autouser:
        logger.info(f'[{self.dbname} Setup]: Creating Auto User Schema(s)')
        if self.config.userconfigs:
            for name, schema_config in self.config.userconfigs.items():
                schema = LazyUserSchema.get_schema(schema_config, is_dev=self.config.is_dev)
                self._db[name] = LazyDBModel(name, schema, LazyUserSchema.get_hash_schema(), is_dev=self.config.is_dev)
                self._base_schemas[name] = self._db[name].schema
                logger.info(f'[{self.dbname} Setup]: Created Custom User Schema: {name}')
        else:
            schema = LazyUserSchema.get_schema(is_dev=self.config.is_dev)
            self._db['user'] = LazyDBModel('user', schema, LazyUserSchema.get_hash_schema(), is_dev=self.config.is_dev)
            self._base_schemas['user'] = self._db['user'].schema
            logger.info(f'[{self.dbname} Setup]: Created Default User Schema')
    for name, schema in self.config.dbschema.items():
        hashschema = self.config.hashschema.get(name, None) if self.config.hashschema else None
        self._db[name] = LazyDBModel(name, schema, hashschema, is_dev=self.config.is_dev)
        self._base_schemas[name] = self._db[name].schema
        logger.info(f'[{self.dbname} Setup]: Created DB Schema Added for: {name}')