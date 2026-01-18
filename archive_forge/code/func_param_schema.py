import json
import textwrap
@classmethod
def param_schema(cls, ptype, p, safe=False, subset=None):
    if ptype in cls.unserializable_parameter_types:
        raise UnserializableException
    dispatch_method = cls._get_method(ptype, 'schema')
    if dispatch_method:
        schema = dispatch_method(p, safe=safe)
    else:
        schema = {'type': ptype.lower()}
    return JSONNullable(schema) if p.allow_None else schema