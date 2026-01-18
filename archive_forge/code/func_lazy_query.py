from __future__ import annotations
from typing import TYPE_CHECKING, Iterator, List, Optional
from langchain_core.utils import get_from_env
def lazy_query(self, query: str) -> Iterator[dict]:
    with self.client.execute_sql(query).open_reader() as reader:
        if reader.count == 0:
            raise ValueError('Table contains no data.')
        for record in reader:
            yield {k: v for k, v in record}