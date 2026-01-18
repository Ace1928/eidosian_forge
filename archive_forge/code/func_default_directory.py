from __future__ import print_function
from __future__ import unicode_literals
import collections
import contextlib
import gzip
import json
import keyword
import logging
import os
import re
import tempfile
import six
from six.moves import urllib_parse
import six.moves.urllib.error as urllib_error
import six.moves.urllib.request as urllib_request
@property
def default_directory(self):
    return self.package