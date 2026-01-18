from __future__ import annotations
import os
import typing
from pathlib import Path
def get_openapi_spec() -> Spec:
    """Get the OpenAPI spec object."""
    from openapi_core.spec.paths import Spec
    openapi_spec_dict = get_openapi_spec_dict()
    return Spec.from_dict(openapi_spec_dict)