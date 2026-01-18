import re
from typing import Dict, Iterator, List
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
@staticmethod
def parse_document_body(body: str) -> str:
    result = re.sub('<a name="(.*)"></a>', '', body)
    result = re.sub('<br\\s*/?>', '', result)
    return result