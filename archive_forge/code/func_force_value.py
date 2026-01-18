import sys
import io
import random
import mimetypes
import time
import os
import shutil
import smtplib
import shlex
import re
import subprocess
from urllib.parse import urlencode
from urllib import parse as urlparse
from http.cookies import BaseCookie
from paste import wsgilib
from paste import lint
from paste.response import HeaderDict
def force_value(self, value):
    """
        Like setting a value, except forces it even for, say, hidden
        fields.
        """
    self._value = value