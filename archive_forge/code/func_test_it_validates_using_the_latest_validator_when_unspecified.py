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
def test_it_validates_using_the_latest_validator_when_unspecified(self):
    self.assertIs(Draft202012Validator, _LATEST_VERSION)
    self.assertOutputs(files=dict(some_schema='{"const": "check"}', some_instance='"a"'), argv=['-i', 'some_instance', 'some_schema'], exit_code=1, stdout='', stderr="a: 'check' was expected\n")