import base64
import os.path
import uuid
from .. import __version__
def prepend_url(url, prefix):
    return '/' + prefix.strip('/') + url