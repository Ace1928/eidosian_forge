import errno
import os
import socket
import struct
import sys
import traceback
import warnings
from . import _auth
from .charset import charset_by_name, charset_by_id
from .constants import CLIENT, COMMAND, CR, ER, FIELD_TYPE, SERVER_STATUS
from . import converters
from .cursors import Cursor
from .optionfile import Parser
from .protocol import (
from . import err, VERSION_STRING
def set_character_set(self, charset, collation=None):
    """
        Set charaset (and collation)

        Send "SET NAMES charset [COLLATE collation]" query.
        Update Connection.encoding based on charset.
        """
    encoding = charset_by_name(charset).encoding
    if collation:
        query = f'SET NAMES {charset} COLLATE {collation}'
    else:
        query = f'SET NAMES {charset}'
    self._execute_command(COMMAND.COM_QUERY, query)
    self._read_packet()
    self.charset = charset
    self.encoding = encoding
    self.collation = collation