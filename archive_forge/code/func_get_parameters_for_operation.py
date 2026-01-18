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
def get_parameters_for_operation(self, operation: Operation) -> List[Parameter]:
    """Get the components for a given operation."""
    from openapi_pydantic import Reference
    parameters = []
    if operation.parameters:
        for parameter in operation.parameters:
            if isinstance(parameter, Reference):
                parameter = self._get_root_referenced_parameter(parameter)
            parameters.append(parameter)
    return parameters