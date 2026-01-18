import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def rot13(chunks, context=None):
    return [codecs.encode(chunk.decode('ascii'), 'rot13').encode('ascii') for chunk in chunks]