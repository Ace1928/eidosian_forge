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
def submit_fields(self, name=None, index=None):
    """
        Return a list of ``[(name, value), ...]`` for the current
        state of the form.
        """
    submit = []
    if name is not None:
        field = self.get(name, index=index)
        submit.append((field.name, field.value_if_submitted()))
    for name, fields in self.fields.items():
        if name is None:
            continue
        for field in fields:
            value = field.value
            if value is None:
                continue
            submit.append((name, value))
    return submit