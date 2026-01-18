import json
import textwrap
@classmethod
def range_schema(cls, p, safe=False):
    schema = cls.tuple_schema(p, safe=safe)
    bounded_number = cls.declare_numeric_bounds({'type': 'number'}, p.bounds, p.inclusive_bounds)
    schema['additionalItems'] = bounded_number
    return schema