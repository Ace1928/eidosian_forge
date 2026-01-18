from __future__ import annotations
import os
import typing
from pathlib import Path
def get_openapi_spec_dict() -> dict[str, typing.Any]:
    """Get the OpenAPI spec as a dictionary."""
    from ruamel.yaml import YAML
    path = HERE / 'rest-api.yml'
    yaml = YAML(typ='safe')
    return yaml.load(path.read_text(encoding='utf-8'))