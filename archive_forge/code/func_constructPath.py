import os
from .constants import YowConstants
import codecs, sys
import logging
import tempfile
import base64
import hashlib
import os.path, mimetypes
import uuid
from consonance.structs.keypair import KeyPair
from appdirs import user_config_dir
from .optionalmodules import PILOptionalModule, FFVideoOptionalModule
@staticmethod
def constructPath(*path):
    path = os.path.join(*path)
    fullPath = os.path.join(user_config_dir(YowConstants.YOWSUP), path)
    if not os.path.exists(os.path.dirname(fullPath)):
        os.makedirs(os.path.dirname(fullPath))
    return fullPath