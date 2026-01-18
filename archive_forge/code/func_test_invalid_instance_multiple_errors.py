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
def test_invalid_instance_multiple_errors(self):
    instance = 12
    first = ValidationError('First error', instance=instance)
    second = ValidationError('Second error', instance=instance)
    self.assertOutputs(files=dict(some_schema='{"does not": "matter since it is stubbed"}', some_instance=json.dumps(instance)), validator=fake_validator([first, second]), argv=['-i', 'some_instance', 'some_schema'], exit_code=1, stderr='                12: First error\n                12: Second error\n            ')