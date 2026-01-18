import logging
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import requests
from langchain_core.documents import Document
from tenacity import (
from langchain_community.document_loaders.base import BaseLoader
def process_attachment(self, page_id: str, ocr_languages: Optional[str]=None) -> List[str]:
    try:
        from PIL import Image
    except ImportError:
        raise ImportError('`Pillow` package not found, please run `pip install Pillow`')
    attachments = self.confluence.get_attachments_from_content(page_id)['results']
    texts = []
    for attachment in attachments:
        media_type = attachment['metadata']['mediaType']
        absolute_url = self.base_url + attachment['_links']['download']
        title = attachment['title']
        try:
            if media_type == 'application/pdf':
                text = title + self.process_pdf(absolute_url, ocr_languages)
            elif media_type == 'image/png' or media_type == 'image/jpg' or media_type == 'image/jpeg':
                text = title + self.process_image(absolute_url, ocr_languages)
            elif media_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                text = title + self.process_doc(absolute_url)
            elif media_type == 'application/vnd.ms-excel':
                text = title + self.process_xls(absolute_url)
            elif media_type == 'image/svg+xml':
                text = title + self.process_svg(absolute_url, ocr_languages)
            else:
                continue
            texts.append(text)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                print(f'Attachment not found at {absolute_url}')
                continue
            else:
                raise
    return texts