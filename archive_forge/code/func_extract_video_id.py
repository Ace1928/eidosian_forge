from __future__ import annotations
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import parse_qs, urlparse
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.pydantic_v1.dataclasses import dataclass
from langchain_community.document_loaders.base import BaseLoader
@staticmethod
def extract_video_id(youtube_url: str) -> str:
    """Extract video id from common YT urls."""
    video_id = _parse_video_id(youtube_url)
    if not video_id:
        raise ValueError(f'Could not determine the video ID for the URL {youtube_url}')
    return video_id