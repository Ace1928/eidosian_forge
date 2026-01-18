import itertools
import logging
import sys
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def get_num_rows(self) -> Tuple[int, int]:
    """Gets the number of "feasible" rows for the DataFrame"""
    try:
        import psutil
    except ImportError as e:
        raise ImportError('psutil not installed. Please install it with `pip install psutil`.') from e
    row = self.df.limit(1).collect()[0]
    estimated_row_size = sys.getsizeof(row)
    mem_info = psutil.virtual_memory()
    available_memory = mem_info.available
    max_num_rows = int(available_memory / estimated_row_size * self.fraction_of_memory)
    return (min(max_num_rows, self.df.count()), max_num_rows)