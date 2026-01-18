import json
import logging
import os
import uuid
from http import HTTPStatus
from typing import Any, Dict, Iterator, List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.pebblo import (
@classmethod
def set_loader_sent(cls) -> None:
    cls._loader_sent = True