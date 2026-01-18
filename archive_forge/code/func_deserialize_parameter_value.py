import json
import textwrap
@classmethod
def deserialize_parameter_value(cls, pobj, pname, value):
    value = cls.loads(value)
    return pobj.param[pname].deserialize(value)