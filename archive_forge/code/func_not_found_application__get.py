import html
import re
import os
from collections.abc import MutableMapping as DictMixin
from paste import httpexceptions
def not_found_application__get(self):
    return self.map.not_found_application