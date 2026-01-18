from importlib import metadata
from json import JSONDecodeError
from textwrap import dedent
import argparse
import json
import sys
import traceback
import warnings
from attrs import define, field
from jsonschema.exceptions import SchemaError
from jsonschema.validators import _RefResolver, validator_for
def parsing_error(self, path, exc_info):
    return 'Failed to parse {}: {}\n'.format('<stdin>' if path == '<stdin>' else repr(path), exc_info[1])