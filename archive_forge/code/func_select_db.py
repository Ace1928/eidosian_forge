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
def select_db(self, db):
    """
        Set current db.

        :param db: The name of the db.
        """
    self._execute_command(COMMAND.COM_INIT_DB, db)
    self._read_ok_packet()