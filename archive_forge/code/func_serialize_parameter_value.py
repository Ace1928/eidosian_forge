import json
import textwrap
@classmethod
def serialize_parameter_value(cls, pobj, pname):
    value = pobj.param.get_value_generator(pname)
    return cls.dumps(pobj.param[pname].serialize(value))