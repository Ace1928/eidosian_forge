import os
import string
import pytest
from .util import random_string
from keyring import errors
def test_delete_not_present(self):
    username = random_string(20, self.DIFFICULT_CHARS)
    service = random_string(20, self.DIFFICULT_CHARS)
    with pytest.raises(errors.PasswordDeleteError):
        self.keyring.delete_password(service, username)