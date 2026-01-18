from __future__ import annotations
import copy
import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
import requests
import yaml
from langchain_core.pydantic_v1 import ValidationError
@classmethod
def from_spec_dict(cls, spec_dict: dict) -> OpenAPISpec:
    """Get an OpenAPI spec from a dict."""
    return cls.parse_obj(spec_dict)