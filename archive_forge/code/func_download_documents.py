import json
from typing import List
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
def download_documents(self, query: str) -> List[Document]:
    """Query the Brave search engine and return the results as a list of Documents.

        Args:
            query: The query to search for.

        Returns: The results as a list of Documents.

        """
    results = self._search_request(query)
    return [Document(page_content=item.get('description'), metadata={'title': item.get('title'), 'link': item.get('url')}) for item in results]