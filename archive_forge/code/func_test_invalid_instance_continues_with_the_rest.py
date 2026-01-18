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
def test_invalid_instance_continues_with_the_rest(self):
    self.assertOutputs(files=dict(some_schema='{"minimum": 30}', first_instance='not valid JSON!', second_instance='12'), argv=['-i', 'first_instance', '-i', 'second_instance', 'some_schema'], exit_code=1, stderr="                Failed to parse 'first_instance': {}\n                12: 12 is less than the minimum of 30\n            ".format(_message_for('not valid JSON!')))