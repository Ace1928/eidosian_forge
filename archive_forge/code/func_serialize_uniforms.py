import numbers
import math
from ipywidgets import Widget, widget_serialization
def serialize_uniforms(uniforms, obj):
    """Serialize a uniform dict"""
    serialized = {}
    for name, uniform in uniforms.items():
        serialized[name] = _serialize_item(uniform['value'])
    return serialized