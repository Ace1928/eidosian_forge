import os
import string
import pytest
from .util import random_string
from keyring import errors
def test_delete_one_in_group(self):
    username1 = random_string(20, self.DIFFICULT_CHARS)
    username2 = random_string(20, self.DIFFICULT_CHARS)
    password = random_string(20, self.DIFFICULT_CHARS)
    service = random_string(20, self.DIFFICULT_CHARS)
    self.keyring.set_password(service, username1, password)
    self.set_password(service, username2, password)
    self.keyring.delete_password(service, username1)
    assert self.keyring.get_password(service, username2) == password