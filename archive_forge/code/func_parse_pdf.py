import os
from fastapi import (
from pydantic import BaseModel
from typing import Iterator
def parse_pdf(blob: Blob) -> Iterator[Document]:
    import fitz
    with blob.as_bytes_io() as stream:
        doc = fitz.Document(stream=stream)
        yield from [Document(page_content=page.get_text(), metadata=dict({'source': blob.source, 'file_path': blob.source, 'page': page.number, 'total_pages': len(doc)}, **{k: doc.metadata[k] for k in doc.metadata if type(doc.metadata[k]) in [str, int]})) for page in doc]