import json
import textwrap
@classmethod
def integer_schema(cls, p, safe=False):
    return cls.number_schema(p)