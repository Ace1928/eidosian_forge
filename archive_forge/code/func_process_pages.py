import logging
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import requests
from langchain_core.documents import Document
from tenacity import (
from langchain_community.document_loaders.base import BaseLoader
def process_pages(self, pages: List[dict], include_restricted_content: bool, include_attachments: bool, include_comments: bool, content_format: ContentFormat, ocr_languages: Optional[str]=None, keep_markdown_format: Optional[bool]=False, keep_newlines: bool=False) -> Iterator[Document]:
    """Process a list of pages into a list of documents."""
    for page in pages:
        if not include_restricted_content and (not self.is_public_page(page)):
            continue
        yield self.process_page(page, include_attachments, include_comments, content_format, ocr_languages=ocr_languages, keep_markdown_format=keep_markdown_format, keep_newlines=keep_newlines)