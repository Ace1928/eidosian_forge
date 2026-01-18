from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_ordered_dict(self, data):
    data_type = type(data)
    tag = 'tag:yaml.org,2002:python/object/apply:%s.%s' % (data_type.__module__, data_type.__name__)
    items = [[key, value] for key, value in data.items()]
    return self.represent_sequence(tag, [items])