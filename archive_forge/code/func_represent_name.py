from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_name(self, data):
    name = '%s.%s' % (data.__module__, data.__name__)
    return self.represent_scalar('tag:yaml.org,2002:python/name:' + name, '')