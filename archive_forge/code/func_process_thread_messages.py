import logging
import re
import xml.etree.cElementTree
import xml.sax.saxutils
from io import BytesIO
from typing import List, Optional, Sequence
from xml.etree.ElementTree import ElementTree
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def process_thread_messages(self, thread_id: str) -> str:
    max_created_usec = None
    messages = []
    while True:
        chunk = self.quip_client.get_messages(thread_id, max_created_usec=max_created_usec, count=100)
        messages.extend(chunk)
        if chunk:
            max_created_usec = chunk[-1]['created_usec'] - 1
        else:
            break
    messages.reverse()
    texts = [message['text'] for message in messages]
    return '\n'.join(texts)