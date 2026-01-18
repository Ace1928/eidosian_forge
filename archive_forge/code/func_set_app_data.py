import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_app_data(self, data):
    """
        Set application data

        :param data: The application data
        :return: None
        """
    self._app_data = data