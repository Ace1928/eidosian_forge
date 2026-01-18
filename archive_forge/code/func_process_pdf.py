import logging
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import requests
from langchain_core.documents import Document
from tenacity import (
from langchain_community.document_loaders.base import BaseLoader
def process_pdf(self, link: str, ocr_languages: Optional[str]=None) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
    except ImportError:
        raise ImportError('`pytesseract` or `pdf2image` package not found, please run `pip install pytesseract pdf2image`')
    response = self.confluence.request(path=link, absolute=True)
    text = ''
    if response.status_code != 200 or response.content == b'' or response.content is None:
        return text
    try:
        images = convert_from_bytes(response.content)
    except ValueError:
        return text
    for i, image in enumerate(images):
        image_text = pytesseract.image_to_string(image, lang=ocr_languages)
        text += f'Page {i + 1}:\n{image_text}\n\n'
    return text