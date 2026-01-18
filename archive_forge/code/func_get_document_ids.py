import re
from typing import Dict, Iterator, List
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def get_document_ids(self, book_id: int) -> List[int]:
    url = f'{self.api_url}/api/v2/repos/{book_id}/docs'
    response = self.http_get(url=url)
    return [document['id'] for document in response['data']]