import inspect
import json
import re
from typing import Callable, Optional
from jsonschema.protocols import Validator
from pydantic import create_model
from referencing import Registry, Resource
from referencing._core import Resolver
from referencing.jsonschema import DRAFT202012
def build_regex_from_schema(schema: str, whitespace_pattern: Optional[str]=None):
    """Turn a JSON schema into a regex that matches any JSON object that follows
    this schema.

    JSON Schema is a declarative language that allows to annotate JSON documents
    with types and descriptions. These schemas can be generated from any Python
    datastructure that has type annotation: namedtuples, dataclasses, Pydantic
    models. And by ensuring that the generation respects the schema we ensure
    that the output can be parsed into these objects.
    This function parses the provided schema and builds a generation schedule which
    mixes deterministic generation (fixed strings), and sampling with constraints.

    Parameters
    ----------
    schema
        A string that represents a JSON Schema.
    whitespace_pattern
        Pattern to use for JSON syntactic whitespace (doesn't impact string literals)
        Example: allow only a single space or newline with `whitespace_pattern=r"[
 ]?"`

    Returns
    -------
    A generation schedule. A list of strings that represent the JSON
    schema's structure and regular expression that define the structure of
    the fields.

    References
    ----------
    .. [0] JSON Schema. https://json-schema.org/

    """
    schema = json.loads(schema)
    Validator.check_schema(schema)
    schema = Resource(contents=schema, specification=DRAFT202012)
    uri = schema.id() if schema.id() is not None else ''
    registry = Registry().with_resource(uri=uri, resource=schema)
    resolver = registry.resolver()
    content = schema.contents
    return to_regex(resolver, content, whitespace_pattern)