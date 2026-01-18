import os
import string
import sys
import unittest
from collections import defaultdict
from unittest.mock import patch
from uuid import UUID
from uuid import uuid4
from shortuuid.cli import cli
from shortuuid.main import decode
from shortuuid.main import encode
from shortuuid.main import get_alphabet
from shortuuid.main import random
from shortuuid.main import set_alphabet
from shortuuid.main import ShortUUID
from shortuuid.main import uuid
def test_decoding(self):
    su = ShortUUID()
    random_uid = uuid4()
    smallest_uid = UUID(int=0)
    encoded_random = su.encode(random_uid)
    encoded_small = su.encode(smallest_uid)
    self.assertEqual(su.decode(encoded_small), smallest_uid)
    self.assertEqual(su.decode(encoded_random), random_uid)