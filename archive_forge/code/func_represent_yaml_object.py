from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_yaml_object(self, tag, data, cls, flow_style=None):
    if hasattr(data, '__getstate__'):
        state = data.__getstate__()
    else:
        state = data.__dict__.copy()
    return self.represent_mapping(tag, state, flow_style=flow_style)