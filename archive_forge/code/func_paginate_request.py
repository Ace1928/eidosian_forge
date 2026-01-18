import logging
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import requests
from langchain_core.documents import Document
from tenacity import (
from langchain_community.document_loaders.base import BaseLoader
def paginate_request(self, retrieval_method: Callable, **kwargs: Any) -> List:
    """Paginate the various methods to retrieve groups of pages.

        Unfortunately, due to page size, sometimes the Confluence API
        doesn't match the limit value. If `limit` is >100 confluence
        seems to cap the response to 100. Also, due to the Atlassian Python
        package, we don't get the "next" values from the "_links" key because
        they only return the value from the result key. So here, the pagination
        starts from 0 and goes until the max_pages, getting the `limit` number
        of pages with each request. We have to manually check if there
        are more docs based on the length of the returned list of pages, rather than
        just checking for the presence of a `next` key in the response like this page
        would have you do:
        https://developer.atlassian.com/server/confluence/pagination-in-the-rest-api/

        :param retrieval_method: Function used to retrieve docs
        :type retrieval_method: callable
        :return: List of documents
        :rtype: List
        """
    max_pages = kwargs.pop('max_pages')
    docs: List[dict] = []
    while len(docs) < max_pages:
        get_pages = retry(reraise=True, stop=stop_after_attempt(self.number_of_retries), wait=wait_exponential(multiplier=1, min=self.min_retry_seconds, max=self.max_retry_seconds), before_sleep=before_sleep_log(logger, logging.WARNING))(retrieval_method)
        batch = get_pages(**kwargs, start=len(docs))
        if not batch:
            break
        docs.extend(batch)
    return docs[:max_pages]