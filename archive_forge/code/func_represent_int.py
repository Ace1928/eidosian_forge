from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_int(self, data):
    return self.represent_scalar('tag:yaml.org,2002:int', str(data))