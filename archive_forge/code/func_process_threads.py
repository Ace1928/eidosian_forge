import logging
import re
import xml.etree.cElementTree
import xml.sax.saxutils
from io import BytesIO
from typing import List, Optional, Sequence
from xml.etree.ElementTree import ElementTree
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def process_threads(self, thread_ids: Sequence[str], include_images: bool, include_messages: bool) -> List[Document]:
    """Process a list of thread into a list of documents."""
    docs = []
    for thread_id in thread_ids:
        doc = self.process_thread(thread_id, include_images, include_messages)
        if doc is not None:
            docs.append(doc)
    return docs