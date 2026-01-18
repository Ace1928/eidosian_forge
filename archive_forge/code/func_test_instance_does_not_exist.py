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
def test_instance_does_not_exist(self):
    self.assertOutputs(files=dict(some_schema='{}'), argv=['-i', 'nonexisting_instance', 'some_schema'], exit_code=1, stderr="                'nonexisting_instance' does not exist.\n            ")