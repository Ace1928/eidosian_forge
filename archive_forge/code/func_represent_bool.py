from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_bool(self, data):
    if data:
        value = 'true'
    else:
        value = 'false'
    return self.represent_scalar('tag:yaml.org,2002:bool', value)