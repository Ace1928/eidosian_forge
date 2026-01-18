import importlib
from django.apps import apps
from django.conf import settings
from django.core.serializers.base import SerializerDoesNotExist
def get_deserializer(format):
    if not _serializers:
        _load_serializers()
    if format not in _serializers:
        raise SerializerDoesNotExist(format)
    return _serializers[format].Deserializer