from typing import List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
def load_device(self, url_override: Optional[str]=None, include_guides: bool=True) -> List[Document]:
    """Loads a device

        Args:
            url_override: A URL to override the default URL.
            include_guides: Whether to include guides linked to from the device.
              Defaults to True.

        Returns:

        """
    documents = []
    if url_override is None:
        url = IFIXIT_BASE_URL + '/wikis/CATEGORY/' + self.id
    else:
        url = url_override
    res = requests.get(url)
    data = res.json()
    text = '\n'.join([data[key] for key in ['title', 'description', 'contents_raw'] if key in data]).strip()
    metadata = {'source': self.web_path, 'title': data['title']}
    documents.append(Document(page_content=text, metadata=metadata))
    if include_guides:
        'Load and return documents for each guide linked to from the device'
        guide_urls = [guide['url'] for guide in data['guides']]
        for guide_url in guide_urls:
            documents.append(IFixitLoader(guide_url).load()[0])
    return documents