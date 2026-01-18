from .charset import MBLENGTH
from .constants import FIELD_TYPE, SERVER_STATUS
from . import err
import struct
import sys
def get_column_length(self):
    if self.type_code == FIELD_TYPE.VAR_STRING:
        mblen = MBLENGTH.get(self.charsetnr, 1)
        return self.length // mblen
    return self.length