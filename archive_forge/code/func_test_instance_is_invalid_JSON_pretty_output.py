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
def test_instance_is_invalid_JSON_pretty_output(self):
    stdout, stderr = self.run_cli(files=dict(some_schema='{}', some_instance='not valid JSON!'), argv=['--output', 'pretty', '-i', 'some_instance', 'some_schema'], exit_code=1)
    self.assertFalse(stdout)
    self.assertIn('(some_instance)===\n\nTraceback (most recent call last):\n', stderr)
    self.assertNotIn('some_schema', stderr)