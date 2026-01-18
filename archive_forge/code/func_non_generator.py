import concurrent
import logging
import random
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence, Type, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.html_bs import BSHTMLLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
def non_generator(item: Path, path: Path, pbar: Optional[Any]) -> List:
    return [x for x in func(item, path, pbar)]