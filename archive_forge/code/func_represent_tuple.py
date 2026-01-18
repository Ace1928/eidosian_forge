from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_tuple(self, data):
    return self.represent_sequence('tag:yaml.org,2002:python/tuple', data)