import logging
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import requests
from langchain_core.documents import Document
from tenacity import (
from langchain_community.document_loaders.base import BaseLoader
def process_svg(self, link: str, ocr_languages: Optional[str]=None) -> str:
    try:
        import pytesseract
        from PIL import Image
        from reportlab.graphics import renderPM
        from svglib.svglib import svg2rlg
    except ImportError:
        raise ImportError('`pytesseract`, `Pillow`, `reportlab` or `svglib` package not found, please run `pip install pytesseract Pillow reportlab svglib`')
    response = self.confluence.request(path=link, absolute=True)
    text = ''
    if response.status_code != 200 or response.content == b'' or response.content is None:
        return text
    drawing = svg2rlg(BytesIO(response.content))
    img_data = BytesIO()
    renderPM.drawToFile(drawing, img_data, fmt='PNG')
    img_data.seek(0)
    image = Image.open(img_data)
    return pytesseract.image_to_string(image, lang=ocr_languages)