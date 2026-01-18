import json
from typing import Any, Dict, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
def get_ticker_news(self, ticker: str) -> Optional[dict]:
    """
        Get the most recent news articles relating to a stock ticker symbol,
        including a summary of the article and a link to the original source.

        /v2/reference/news
        """
    url = f'{POLYGON_BASE_URL}v2/reference/news?ticker={ticker}&apiKey={self.polygon_api_key}'
    response = requests.get(url)
    data = response.json()
    status = data.get('status', None)
    if status != 'OK':
        raise ValueError(f'API Error: {data}')
    return data.get('results', None)