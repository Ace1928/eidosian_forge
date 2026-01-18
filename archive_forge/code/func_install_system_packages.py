from __future__ import annotations
import ast
import json
import os
from io import StringIO
from sys import version_info
from typing import IO, TYPE_CHECKING, Any, Callable, List, Optional, Type, Union
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain_community.tools import BaseTool, Tool
from langchain_community.tools.e2b_data_analysis.unparse import Unparser
def install_system_packages(self, package_names: Union[str, List[str]]) -> None:
    """Install system packages (via apt) in the sandbox."""
    self.session.install_system_packages(package_names)