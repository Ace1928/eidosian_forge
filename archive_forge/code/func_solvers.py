import io
import os
import re
import sys
import time
import socket
import base64
import tempfile
import logging
from pyomo.common.dependencies import attempt_import
def solvers(self):
    if self.neos is None:
        return []
    else:
        attempt = 0
        while attempt < 3:
            try:
                return self.neos.listSolversInCategory('kestrel')
            except socket.timeout:
                attempt += 1
        return []