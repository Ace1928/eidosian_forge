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
def generatePreviewFromVideo(videoFile):
    with FFVideoOptionalModule() as imp:
        VideoStream = imp('VideoStream')
        fd, path = tempfile.mkstemp('.jpg')
        stream = VideoStream(videoFile)
        stream.get_frame_at_sec(0).image().save(path)
        preview = ImageTools.generatePreviewFromImage(path)
        os.remove(path)
        return preview