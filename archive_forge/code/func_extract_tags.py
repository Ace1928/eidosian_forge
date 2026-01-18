from typing import Any, Iterator, List, Sequence, cast
from langchain_core.documents import BaseDocumentTransformer, Document
@staticmethod
def extract_tags(html_content: str, tags: List[str], *, remove_comments: bool=False) -> str:
    """
        Extract specific tags from a given HTML content.

        Args:
            html_content: The original HTML content string.
            tags: A list of tags to be extracted from the HTML.

        Returns:
            A string combining the content of the extracted tags.
        """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    text_parts: List[str] = []
    for element in soup.find_all():
        if element.name in tags:
            text_parts += get_navigable_strings(element, remove_comments=remove_comments)
            element.decompose()
    return ' '.join(text_parts)