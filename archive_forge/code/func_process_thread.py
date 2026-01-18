import logging
import re
import xml.etree.cElementTree
import xml.sax.saxutils
from io import BytesIO
from typing import List, Optional, Sequence
from xml.etree.ElementTree import ElementTree
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def process_thread(self, thread_id: str, include_images: bool, include_messages: bool) -> Optional[Document]:
    thread = self.quip_client.get_thread(thread_id)
    thread_id = thread['thread']['id']
    title = thread['thread']['title']
    link = thread['thread']['link']
    update_ts = thread['thread']['updated_usec']
    sanitized_title = QuipLoader._sanitize_title(title)
    logger.info(f'processing thread {thread_id} title {sanitized_title} link {link} update_ts {update_ts}')
    if 'html' in thread:
        try:
            tree = self.quip_client.parse_document_html(thread['html'])
        except xml.etree.cElementTree.ParseError as e:
            logger.error(f'Error parsing thread {title} {thread_id}, skipping, {e}')
            return None
        metadata = {'title': sanitized_title, 'update_ts': update_ts, 'id': thread_id, 'source': link}
        text = ''
        if include_images:
            text = self.process_thread_images(tree)
        if include_messages:
            text = text + '/n' + self.process_thread_messages(thread_id)
        return Document(page_content=thread['html'] + text, metadata=metadata)
    return None