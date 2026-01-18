from functools import lru_cache
from django.db.backends.base.base import NO_DB_ALIAS
from django.db.backends.postgresql.base import DatabaseWrapper as PsycopgDatabaseWrapper
from django.db.backends.postgresql.features import (
from django.db.backends.postgresql.introspection import (
from django.db.backends.postgresql.operations import (
from django.db.backends.postgresql.psycopg_any import is_psycopg3
from .adapter import PostGISAdapter
from .features import DatabaseFeatures
from .introspection import PostGISIntrospection
from .operations import PostGISOperations
from .schema import PostGISSchemaEditor
@lru_cache
def postgis_adapters(geo_oid, geog_oid, raster_oid):

    class BaseDumper(Dumper):

        def __init_subclass__(cls, base_dumper):
            super().__init_subclass__()
            cls.GeometryDumper = type('GeometryDumper', (base_dumper,), {'oid': geo_oid})
            cls.GeographyDumper = type('GeographyDumper', (base_dumper,), {'oid': geog_oid})
            cls.RasterDumper = type('RasterDumper', (BaseTextDumper,), {'oid': raster_oid})

        def get_key(self, obj, format):
            if obj.is_geometry:
                return GeographyType if obj.geography else GeometryType
            else:
                return RasterType

        def upgrade(self, obj, format):
            if obj.is_geometry:
                if obj.geography:
                    return self.GeographyDumper(GeographyType)
                else:
                    return self.GeometryDumper(GeometryType)
            else:
                return self.RasterDumper(RasterType)

        def dump(self, obj):
            raise NotImplementedError

    class PostGISTextDumper(BaseDumper, base_dumper=BaseTextDumper):
        pass

    class PostGISBinaryDumper(BaseDumper, base_dumper=BaseBinaryDumper):
        format = Format.BINARY
    return (PostGISTextDumper, PostGISBinaryDumper)