import collections
import decimal
from io import StringIO
import yaml
from django.core.serializers.base import DeserializationError
from django.core.serializers.python import Deserializer as PythonDeserializer
from django.core.serializers.python import Serializer as PythonSerializer
from django.db import models
def represent_decimal(self, data):
    return self.represent_scalar('tag:yaml.org,2002:str', str(data))