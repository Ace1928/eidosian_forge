import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
def getRecordedProperties(self):
    """Return any properties that the user has recorded."""
    return self.__recorded_properties