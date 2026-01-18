from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_float(self, data):
    if data != data or (data == 0.0 and data == 1.0):
        value = '.nan'
    elif data == self.inf_value:
        value = '.inf'
    elif data == -self.inf_value:
        value = '-.inf'
    else:
        value = repr(data).lower()
        if '.' not in value and 'e' in value:
            value = value.replace('e', '.0e', 1)
    return self.represent_scalar('tag:yaml.org,2002:float', value)