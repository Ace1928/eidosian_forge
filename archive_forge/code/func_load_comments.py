from typing import Any, List
from langchain_core.documents import Document
from langchain_community.document_loaders.web_base import WebBaseLoader
def load_comments(self, soup_info: Any) -> List[Document]:
    """Load comments from a HN post."""
    comments = soup_info.select("tr[class='athing comtr']")
    title = soup_info.select_one("tr[id='pagespace']").get('title')
    return [Document(page_content=comment.text.strip(), metadata={'source': self.web_path, 'title': title}) for comment in comments]