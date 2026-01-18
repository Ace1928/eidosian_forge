from __future__ import annotations
import logging
import os
import pathlib
import platform
from typing import Optional, Tuple
from langchain_core.env import get_runtime_environment
from langchain_core.pydantic_v1 import BaseModel
from langchain_community.document_loaders.base import BaseLoader
def get_loader_full_path(loader: BaseLoader) -> str:
    """Return an absolute source path of source of loader based on the
    keys present in Document object from loader.

    Args:
        loader (BaseLoader): Langchain document loader, derived from Baseloader.
    """
    from langchain_community.document_loaders import DataFrameLoader, GCSFileLoader, NotionDBLoader, S3FileLoader
    location = '-'
    if not isinstance(loader, BaseLoader):
        logger.error('loader is not derived from BaseLoader, source location will be unknown!')
        return location
    loader_dict = loader.__dict__
    try:
        if 'bucket' in loader_dict:
            if isinstance(loader, GCSFileLoader):
                location = f'gc://{loader.bucket}/{loader.blob}'
            elif isinstance(loader, S3FileLoader):
                location = f's3://{loader.bucket}/{loader.key}'
        elif 'source' in loader_dict:
            location = loader_dict['source']
            if location and 'channel' in loader_dict:
                channel = loader_dict['channel']
                if channel:
                    location = f'{location}/{channel}'
        elif 'path' in loader_dict:
            location = loader_dict['path']
        elif 'file_path' in loader_dict:
            location = loader_dict['file_path']
        elif 'web_paths' in loader_dict:
            web_paths = loader_dict['web_paths']
            if web_paths and isinstance(web_paths, list) and (len(web_paths) > 0):
                location = web_paths[0]
        elif isinstance(loader, DataFrameLoader):
            location = 'in-memory'
        elif isinstance(loader, NotionDBLoader):
            location = f'notiondb://{loader.database_id}'
    except Exception:
        pass
    return get_full_path(str(location))