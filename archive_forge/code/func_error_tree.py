from __future__ import annotations
import os
import fastjsonschema
import jsonschema
from fastjsonschema import JsonSchemaException as _JsonSchemaException
from jsonschema import Draft4Validator as _JsonSchemaValidator
from jsonschema.exceptions import ErrorTree, ValidationError
def error_tree(self, errors):
    """Create an error tree for the errors."""
    msg = 'JSON schema error introspection not enabled for fastjsonschema'
    raise NotImplementedError(msg)