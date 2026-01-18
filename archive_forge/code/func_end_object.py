import json
from django.core.serializers.base import DeserializationError
from django.core.serializers.json import DjangoJSONEncoder
from django.core.serializers.python import Deserializer as PythonDeserializer
from django.core.serializers.python import Serializer as PythonSerializer
def end_object(self, obj):
    json.dump(self.get_dump_object(obj), self.stream, **self.json_kwargs)
    self.stream.write('\n')
    self._current = None