import sys
from decimal import Decimal
from decimal import InvalidOperation as DecimalInvalidOperation
from pathlib import Path
from django.contrib.gis.db.models import GeometryField
from django.contrib.gis.gdal import (
from django.contrib.gis.gdal.field import (
from django.core.exceptions import FieldDoesNotExist, ObjectDoesNotExist
from django.db import connections, models, router, transaction
from django.utils.encoding import force_str
def check_layer(self):
    """
        Check the Layer metadata and ensure that it's compatible with the
        mapping information and model. Unlike previous revisions, there is no
        need to increment through each feature in the Layer.
        """
    self.geom_field = False
    self.fields = {}
    ogr_fields = self.layer.fields
    ogr_field_types = self.layer.field_types

    def check_ogr_fld(ogr_map_fld):
        try:
            idx = ogr_fields.index(ogr_map_fld)
        except ValueError:
            raise LayerMapError('Given mapping OGR field "%s" not found in OGR Layer.' % ogr_map_fld)
        return idx
    for field_name, ogr_name in self.mapping.items():
        try:
            model_field = self.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            raise LayerMapError('Given mapping field "%s" not in given Model fields.' % field_name)
        fld_name = model_field.__class__.__name__
        if isinstance(model_field, GeometryField):
            if self.geom_field:
                raise LayerMapError('LayerMapping does not support more than one GeometryField per model.')
            coord_dim = model_field.dim
            try:
                if coord_dim == 3:
                    gtype = OGRGeomType(ogr_name + '25D')
                else:
                    gtype = OGRGeomType(ogr_name)
            except GDALException:
                raise LayerMapError('Invalid mapping for GeometryField "%s".' % field_name)
            ltype = self.layer.geom_type
            if not (ltype.name.startswith(gtype.name) or self.make_multi(ltype, model_field)):
                raise LayerMapError('Invalid mapping geometry; model has %s%s, layer geometry type is %s.' % (fld_name, '(dim=3)' if coord_dim == 3 else '', ltype))
            self.geom_field = field_name
            self.coord_dim = coord_dim
            fields_val = model_field
        elif isinstance(model_field, models.ForeignKey):
            if isinstance(ogr_name, dict):
                rel_model = model_field.remote_field.model
                for rel_name, ogr_field in ogr_name.items():
                    idx = check_ogr_fld(ogr_field)
                    try:
                        rel_model._meta.get_field(rel_name)
                    except FieldDoesNotExist:
                        raise LayerMapError('ForeignKey mapping field "%s" not in %s fields.' % (rel_name, rel_model.__class__.__name__))
                fields_val = rel_model
            else:
                raise TypeError('ForeignKey mapping must be of dictionary type.')
        else:
            if model_field.__class__ not in self.FIELD_TYPES:
                raise LayerMapError('Django field type "%s" has no OGR mapping (yet).' % fld_name)
            idx = check_ogr_fld(ogr_name)
            ogr_field = ogr_field_types[idx]
            if not issubclass(ogr_field, self.FIELD_TYPES[model_field.__class__]):
                raise LayerMapError('OGR field "%s" (of type %s) cannot be mapped to Django %s.' % (ogr_field, ogr_field.__name__, fld_name))
            fields_val = model_field
        self.fields[field_name] = fields_val