import json
import textwrap
@classmethod
def class__schema(cls, class_, safe=False):
    from .parameterized import Parameterized
    if isinstance(class_, tuple):
        return {'anyOf': [cls.class__schema(cls_) for cls_ in class_]}
    elif class_ in cls.json_schema_literal_types:
        return {'type': cls.json_schema_literal_types[class_]}
    elif issubclass(class_, Parameterized):
        return {'type': 'object', 'properties': class_.param.schema(safe)}
    else:
        return {'type': 'object'}