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
@classmethod
def generateUUID(cls):
    """
        :return:
        :rtype: uuid.UUID
        """
    return uuid.uuid4()