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
def docai_parse(self, blobs: Sequence[Blob], *, gcs_output_path: Optional[str]=None, processor_name: Optional[str]=None, batch_size: int=1000, enable_native_pdf_parsing: bool=True, field_mask: Optional[str]=None) -> List['Operation']:
    """Runs Google Document AI PDF Batch Processing on a list of blobs.

        Args:
            blobs: a list of blobs to be parsed
            gcs_output_path: a path (folder) on GCS to store results
            processor_name: name of a Document AI processor.
            batch_size: amount of documents per batch
            enable_native_pdf_parsing: a config option for the parser
            field_mask: a comma-separated list of which fields to include in the
                Document AI response.
                suggested: "text,pages.pageNumber,pages.layout"

        Document AI has a 1000 file limit per batch, so batches larger than that need
        to be split into multiple requests.
        Batch processing is an async long-running operation
        and results are stored in a output GCS bucket.
        """
    try:
        from google.cloud import documentai
        from google.cloud.documentai_v1.types import OcrConfig, ProcessOptions
    except ImportError as exc:
        raise ImportError('documentai package not found, please install it with `pip install google-cloud-documentai`') from exc
    output_path = gcs_output_path or self._gcs_output_path
    if output_path is None:
        raise ValueError('An output path on Google Cloud Storage should be provided.')
    processor_name = processor_name or self._processor_name
    if processor_name is None:
        raise ValueError('A Document AI processor name should be provided.')
    operations = []
    for batch in batch_iterate(size=batch_size, iterable=blobs):
        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=documentai.GcsDocuments(documents=[documentai.GcsDocument(gcs_uri=blob.path, mime_type=blob.mimetype or 'application/pdf') for blob in batch]))
        output_config = documentai.DocumentOutputConfig(gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(gcs_uri=output_path, field_mask=field_mask))
        process_options = ProcessOptions(ocr_config=OcrConfig(enable_native_pdf_parsing=enable_native_pdf_parsing)) if enable_native_pdf_parsing else None
        operations.append(self._client.batch_process_documents(documentai.BatchProcessRequest(name=processor_name, input_documents=input_config, document_output_config=output_config, process_options=process_options, skip_human_review=True)))
    return operations