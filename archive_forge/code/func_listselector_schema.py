import json
import textwrap
@classmethod
def listselector_schema(cls, p, safe=False):
    if p.objects is None:
        if safe is True:
            msg = 'ListSelector cannot be guaranteed to be safe for serialization as allowed objects unspecified'
        return {'type': 'array'}
    for obj in p.objects:
        if type(obj) not in cls.json_schema_literal_types:
            msg = 'ListSelector cannot serialize type %s' % type(obj)
            raise UnserializableException(msg)
    return {'type': 'array', 'items': {'enum': p.objects}}