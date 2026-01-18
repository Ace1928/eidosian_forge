from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_list(self, data):
    return self.represent_sequence('tag:yaml.org,2002:seq', data)