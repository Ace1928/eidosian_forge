import base64
import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Type
import requests
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import Tool
def strip_markdown_code(md_string: str) -> str:
    """Strip markdown code from a string."""
    stripped_string = re.sub('^`{1,3}.*?\\n', '', md_string, flags=re.DOTALL)
    stripped_string = re.sub('`{1,3}$', '', stripped_string)
    return stripped_string