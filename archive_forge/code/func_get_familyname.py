from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
def get_familyname(self):
    """Return the font family name, e.g., 'Times'."""
    name = self._header.get(b'FamilyName')
    if name is not None:
        return name
    name = self.get_fullname()
    extras = '(?i)([ -](regular|plain|italic|oblique|bold|semibold|light|ultralight|extra|condensed))+$'
    return re.sub(extras, '', name)