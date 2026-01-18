import atexit
import errno
import os
import re
import shutil
import sys
import tempfile
from hashlib import md5
from io import BytesIO
from json import dumps
from time import sleep
from httplib2 import Http, urlnorm
from wadllib.application import Application
from lazr.restfulclient._json import DatetimeJSONEncoder
from lazr.restfulclient.errors import HTTPError, error_for
from lazr.uri import URI
def ssl_certificate_validation_disabled():
    """Whether the user has disabled SSL certificate connection.

    Some testing servers have broken certificates.  Rather than raising an
    error, we allow an environment variable,
    ``LP_DISABLE_SSL_CERTIFICATE_VALIDATION`` to disable the check.
    """
    return bool(os.environ.get('LP_DISABLE_SSL_CERTIFICATE_VALIDATION', False))