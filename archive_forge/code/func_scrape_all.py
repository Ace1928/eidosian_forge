import asyncio
import logging
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
import aiohttp
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def scrape_all(self, urls: List[str], parser: Union[str, None]=None) -> List[Any]:
    """Fetch all urls, then return soups for all results."""
    from bs4 import BeautifulSoup
    results = asyncio.run(self.fetch_all(urls))
    final_results = []
    for i, result in enumerate(results):
        url = urls[i]
        if parser is None:
            if url.endswith('.xml'):
                parser = 'xml'
            else:
                parser = self.default_parser
            self._check_parser(parser)
        final_results.append(BeautifulSoup(result, parser, **self.bs_kwargs))
    return final_results