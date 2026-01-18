from __future__ import absolute_import, division, print_function
import hashlib
import re
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.six.moves import zip
def get_section_objects(self, path):
    if not isinstance(path, list):
        path = [path]
    obj = self.get_object(path)
    if not obj:
        raise ValueError('path does not exist in config')
    return self.expand_section(obj)