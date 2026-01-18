from functools import partial
import json
import math
import warnings
from fiona.model import Geometry, to_dict
from fiona._vendor.munch import munchify
def make_ld_context(context_items):
    """Returns a JSON-LD Context object.

    See https://json-ld.org/spec/latest/json-ld/."""
    ctx = {'@context': {'geojson': 'http://ld.geojson.org/vocab#', 'Feature': 'geojson:Feature', 'FeatureCollection': 'geojson:FeatureCollection', 'GeometryCollection': 'geojson:GeometryCollection', 'LineString': 'geojson:LineString', 'MultiLineString': 'geojson:MultiLineString', 'MultiPoint': 'geojson:MultiPoint', 'MultiPolygon': 'geojson:MultiPolygon', 'Point': 'geojson:Point', 'Polygon': 'geojson:Polygon', 'bbox': {'@container': '@list', '@id': 'geojson:bbox'}, 'coordinates': 'geojson:coordinates', 'datetime': 'http://www.w3.org/2006/time#inXSDDateTime', 'description': 'http://purl.org/dc/terms/description', 'features': {'@container': '@set', '@id': 'geojson:features'}, 'geometry': 'geojson:geometry', 'id': '@id', 'properties': 'geojson:properties', 'start': 'http://www.w3.org/2006/time#hasBeginning', 'stop': 'http://www.w3.org/2006/time#hasEnding', 'title': 'http://purl.org/dc/terms/title', 'type': '@type', 'when': 'geojson:when'}}
    for item in context_items or []:
        t, uri = item.split('=')
        ctx[t.strip()] = uri.strip()
    return ctx