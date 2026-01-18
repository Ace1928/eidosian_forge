from contextlib import redirect_stderr, redirect_stdout
from importlib import metadata
from io import StringIO
from json import JSONDecodeError
from pathlib import Path
from textwrap import dedent
from unittest import TestCase
import json
import os
import subprocess
import sys
import tempfile
import warnings
from jsonschema import Draft4Validator, Draft202012Validator
from jsonschema.exceptions import (
from jsonschema.validators import _LATEST_VERSION, validate
def test_invalid_instance(self):
    error = ValidationError('I am an error!', instance=12)
    self.assertOutputs(files=dict(some_schema='{"does not": "matter since it is stubbed"}', some_instance=json.dumps(error.instance)), validator=fake_validator([error]), argv=['-i', 'some_instance', 'some_schema'], exit_code=1, stderr='12: I am an error!\n')