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
def all_headers(self, name):
    """
        Gets all headers by the ``name``, returns as a list
        """
    found = []
    for cur_name, value in self.headers:
        if cur_name.lower() == name.lower():
            found.append(value)
    return found