import logging
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import requests
from langchain_core.documents import Document
from tenacity import (
from langchain_community.document_loaders.base import BaseLoader
def process_image(self, link: str, ocr_languages: Optional[str]=None) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise ImportError('`pytesseract` or `Pillow` package not found, please run `pip install pytesseract Pillow`')
    response = self.confluence.request(path=link, absolute=True)
    text = ''
    if response.status_code != 200 or response.content == b'' or response.content is None:
        return text
    try:
        image = Image.open(BytesIO(response.content))
    except OSError:
        return text
    return pytesseract.image_to_string(image, lang=ocr_languages)