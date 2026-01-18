import re
from ovs.db import error
def get_optional(self, name, types, default=None):
    return self.__get(name, types, True, default)