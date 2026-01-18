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
def fake_validator(*errors):
    errors = list(reversed(errors))

    class FakeValidator:

        def __init__(self, *args, **kwargs):
            pass

        def iter_errors(self, instance):
            if errors:
                return errors.pop()
            return []

        @classmethod
        def check_schema(self, schema):
            pass
    return FakeValidator