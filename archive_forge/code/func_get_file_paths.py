import base64
from abc import ABC
from datetime import datetime
from typing import Callable, Dict, Iterator, List, Literal, Optional, Union
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders.base import BaseLoader
def get_file_paths(self) -> List[Dict]:
    base_url = f'{self.github_api_url}/repos/{self.repo}/git/trees/{self.branch}?recursive=1'
    response = requests.get(base_url, headers=self.headers)
    response.raise_for_status()
    all_files = response.json()['tree']
    " one element in all_files\n        {\n            'path': '.github', \n            'mode': '040000', \n            'type': 'tree', \n            'sha': '5dc46e6b38b22707894ced126270b15e2f22f64e', \n            'url': 'https://api.github.com/repos/langchain-ai/langchain/git/blobs/5dc46e6b38b22707894ced126270b15e2f22f64e'\n        }\n        "
    return [f for f in all_files if not (self.file_filter and (not self.file_filter(f['path'])))]