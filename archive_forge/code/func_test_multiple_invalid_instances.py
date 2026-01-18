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
def test_multiple_invalid_instances(self):
    first_instance = 12
    first_errors = [ValidationError('An error', instance=first_instance), ValidationError('Another error', instance=first_instance)]
    second_instance = 'foo'
    second_errors = [ValidationError('BOOM', instance=second_instance)]
    self.assertOutputs(files=dict(some_schema='{"does not": "matter since it is stubbed"}', some_first_instance=json.dumps(first_instance), some_second_instance=json.dumps(second_instance)), validator=fake_validator(first_errors, second_errors), argv=['-i', 'some_first_instance', '-i', 'some_second_instance', 'some_schema'], exit_code=1, stderr='                12: An error\n                12: Another error\n                foo: BOOM\n            ')