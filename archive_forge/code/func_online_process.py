import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.utils.iter import batch_iterate
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.utilities.vertexai import get_client_info
def online_process(self, blob: Blob, enable_native_pdf_parsing: bool=True, field_mask: Optional[str]=None, page_range: Optional[List[int]]=None) -> Iterator[Document]:
    """Parses a blob lazily using online processing.

        Args:
            blob: a blob to parse.
            enable_native_pdf_parsing: enable pdf embedded text extraction
            field_mask: a comma-separated list of which fields to include in the
                Document AI response.
                suggested: "text,pages.pageNumber,pages.layout"
            page_range: list of page numbers to parse. If `None`,
                entire document will be parsed.
        """
    try:
        from google.cloud import documentai
        from google.cloud.documentai_v1.types import IndividualPageSelector, OcrConfig, ProcessOptions
    except ImportError as exc:
        raise ImportError('documentai package not found, please install it with `pip install google-cloud-documentai`') from exc
    try:
        from google.cloud.documentai_toolbox.wrappers.page import _text_from_layout
    except ImportError as exc:
        raise ImportError('documentai_toolbox package not found, please install it with `pip install google-cloud-documentai-toolbox`') from exc
    ocr_config = OcrConfig(enable_native_pdf_parsing=enable_native_pdf_parsing) if enable_native_pdf_parsing else None
    individual_page_selector = IndividualPageSelector(pages=page_range) if page_range else None
    response = self._client.process_document(documentai.ProcessRequest(name=self._processor_name, gcs_document=documentai.GcsDocument(gcs_uri=blob.path, mime_type=blob.mimetype or 'application/pdf'), process_options=ProcessOptions(ocr_config=ocr_config, individual_page_selector=individual_page_selector), skip_human_review=True, field_mask=field_mask))
    yield from (Document(page_content=_text_from_layout(page.layout, response.document.text), metadata={'page': page.page_number, 'source': blob.path}) for page in response.document.pages)