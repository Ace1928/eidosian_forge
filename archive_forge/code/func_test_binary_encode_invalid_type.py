import base64
import hashlib
import hmac
from unittest import mock
import uuid
from osprofiler import _utils as utils
from osprofiler.tests import test
def test_binary_encode_invalid_type(self):
    self.assertRaises(TypeError, utils.binary_encode, 1234)