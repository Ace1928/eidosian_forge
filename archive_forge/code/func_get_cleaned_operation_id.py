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
@staticmethod
def get_cleaned_operation_id(operation: Operation, path: str, method: str) -> str:
    """Get a cleaned operation id from an operation id."""
    operation_id = operation.operationId
    if operation_id is None:
        path = re.sub('[^a-zA-Z0-9]', '_', path.lstrip('/'))
        operation_id = f'{path}_{method}'
    return operation_id.replace('-', '_').replace('.', '_').replace('/', '_')