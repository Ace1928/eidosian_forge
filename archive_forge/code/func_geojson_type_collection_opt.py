import sys
from warnings import warn
import click
from .features import normalize_feature_inputs
def geojson_type_collection_opt(default=False):
    """GeoJSON FeatureCollection output mode"""
    return click.option('--collection', 'geojson_type', flag_value='collection', default=default, help='Output as GeoJSON feature collection(s).')