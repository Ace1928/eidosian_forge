import json
from django.contrib.gis.gdal import CoordTransform, SpatialReference
from django.core.serializers.base import SerializerDoesNotExist
from django.core.serializers.json import Serializer as JSONSerializer
def start_object(self, obj):
    super().start_object(obj)
    self._geometry = None
    if self.geometry_field is None:
        for field in obj._meta.fields:
            if hasattr(field, 'geom_type'):
                self.geometry_field = field.name
                break