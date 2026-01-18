import json
import logging
import os
import re
import tempfile
import time
from abc import ABC
from io import StringIO
from pathlib import Path
from typing import (
from urllib.parse import urlparse
import requests
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import (
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
def get_processed_pdf(self, pdf_id: str) -> str:
    self.wait_for_processing(pdf_id)
    url = f'{self.url}/{pdf_id}.{self.processed_file_format}'
    response = requests.get(url, headers=self._mathpix_headers)
    return response.content.decode('utf-8')