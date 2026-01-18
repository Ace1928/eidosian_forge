from ._base import *
from .types import CreateSchemaType, UpdateSchemaType
from .models import LazyHasher, LazyUserSchema, LazyDBConfig, LazyDBSaveMetrics
def migrate_db(self):
    if not self.config.seeddata:
        logger.info(f'[{self.dbname} Migrate]: No Seed Data Provided. Skipping Migration')
    for schema_name, index in self._db.items():
        if index.idx == 0 and self.config.seeddata.get(schema_name):
            logger.info(f'[DB Migrate]: Running Migration for {schema_name}')
            for item in self.config.seeddata[schema_name]:
                i = self._db[schema_name].create(data=item)
                logger.info(f'[{self.dbname} Migrate]: Created Item [{schema_name}] = ID: {i.uid}, DBID: {i.dbid}')
    logger.info(f'[{self.dbname} Migrate]: Completed all Setup and Migration Tasks')
    self.save_db()