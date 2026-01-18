import json
import textwrap
@classmethod
def objectselector_schema(cls, p, safe=False):
    try:
        allowed_types = [{'type': cls.json_schema_literal_types[type(obj)]} for obj in p.objects]
        schema = {'anyOf': allowed_types}
        schema['enum'] = p.objects
        return schema
    except:
        if safe is True:
            msg = 'ObjectSelector cannot be guaranteed to be safe for serialization due to unserializable type in objects'
            raise UnsafeserializableException(msg)
        return {}