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
def test_it_validates_using_draft4_when_specified(self):
    """
        Specifically, `const` validation *does not* apply for Draft 4.
        """
    schema = '\n            {\n                "$schema": "http://json-schema.org/draft-04/schema#",\n                "const": "check"\n            }\n            '
    instance = '"foo"'
    self.assertOutputs(files=dict(some_schema=schema, some_instance=instance), argv=['-i', 'some_instance', 'some_schema'], stdout='', stderr='')