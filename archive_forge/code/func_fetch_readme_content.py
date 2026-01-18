from typing import Iterator, List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def fetch_readme_content(self, model_id: str) -> str:
    """Fetch the README content for a given model."""
    readme_url = self.README_BASE_URL.format(model_id=model_id)
    try:
        response = requests.get(readme_url)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return 'README not available for this model.'