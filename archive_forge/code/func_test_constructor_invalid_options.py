import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
def test_constructor_invalid_options(self):
    credential_source = {'unsupported': 'value'}
    with pytest.raises(ValueError) as excinfo:
        self.make_pluggable(credential_source=credential_source)
    assert excinfo.match('Missing credential_source')