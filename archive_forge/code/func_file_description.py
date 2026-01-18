import base64
import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Type
import requests
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import Tool
@property
def file_description(self) -> str:
    if len(self.files) == 0:
        return ''
    lines = ['The following files available in the evaluation environment:']
    for target_path, file_info in self.files.items():
        peek_content = head_file(file_info.source_path, 4)
        lines.append(f'- path: `{target_path}` \n first four lines: {peek_content} \n description: `{file_info.description}`')
    return '\n'.join(lines)