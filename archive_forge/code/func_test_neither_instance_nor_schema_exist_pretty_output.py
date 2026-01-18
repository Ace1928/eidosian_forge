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
def test_neither_instance_nor_schema_exist_pretty_output(self):
    self.assertOutputs(argv=['--output', 'pretty', '-i', 'nonexisting_instance', 'nonexisting_schema'], exit_code=1, stderr="                ===[FileNotFoundError]===(nonexisting_schema)===\n\n                'nonexisting_schema' does not exist.\n                -----------------------------\n            ")