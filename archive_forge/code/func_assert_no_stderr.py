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
def assert_no_stderr(self):
    __tracebackhide__ = True
    if self.stderr:
        print('Error output:')
        print(self.stderr)
        raise AssertionError('stderr output not expected')