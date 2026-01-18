from __future__ import annotations
import asyncio
import json
from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator
def is_endpoint_live(url: str, headers: Optional[dict], payload: Any) -> bool:
    """
    Check if an endpoint is live by sending a GET request to the specified URL.

    Args:
        url (str): The URL of the endpoint to check.

    Returns:
        bool: True if the endpoint is live (status code 200), False otherwise.

    Raises:
        Exception: If the endpoint returns a non-successful status code or if there is
            an error querying the endpoint.
    """
    try:
        response = requests.request('POST', url, headers=headers, data=payload)
        if response.status_code == 200:
            return True
        else:
            raise Exception(f'Endpoint returned a non-successful status code: {response.status_code}')
    except requests.exceptions.RequestException as e:
        raise Exception(f'Error querying the endpoint: {e}')