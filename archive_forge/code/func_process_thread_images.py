import logging
import re
import xml.etree.cElementTree
import xml.sax.saxutils
from io import BytesIO
from typing import List, Optional, Sequence
from xml.etree.ElementTree import ElementTree
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def process_thread_images(self, tree: ElementTree) -> str:
    text = ''
    try:
        from PIL import Image
        from pytesseract import pytesseract
    except ImportError:
        raise ImportError('`Pillow or pytesseract` package not found, please run `pip install Pillow` or `pip install pytesseract`')
    for img in tree.iter('img'):
        src = img.get('src')
        if not src or not src.startswith('/blob'):
            continue
        _, _, thread_id, blob_id = src.split('/')
        blob_response = self.quip_client.get_blob(thread_id, blob_id)
        try:
            image = Image.open(BytesIO(blob_response.read()))
            text = text + '\n' + pytesseract.image_to_string(image)
        except OSError as e:
            logger.error(f'failed to convert image to text, {e}')
            raise e
    return text