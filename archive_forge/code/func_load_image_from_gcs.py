from importlib import metadata
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
def load_image_from_gcs(path: str, project: Optional[str]=None) -> 'Image':
    """Load an image from Google Cloud Storage."""
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError('Could not import google-cloud-storage python package.')
    from vertexai.preview.generative_models import Image
    gcs_client = storage.Client(project=project)
    pieces = path.split('/')
    blobs = list(gcs_client.list_blobs(pieces[2], prefix='/'.join(pieces[3:])))
    if len(blobs) > 1:
        raise ValueError(f'Found more than one candidate for {path}!')
    return Image.from_bytes(blobs[0].download_as_bytes())