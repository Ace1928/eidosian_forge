import json
from typing import Any, Dict, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
def get_last_quote(self, ticker: str) -> Optional[dict]:
    """
        Get the most recent National Best Bid and Offer (Quote) for a ticker.

        /v2/last/nbbo/{ticker}
        """
    url = f'{POLYGON_BASE_URL}v2/last/nbbo/{ticker}?apiKey={self.polygon_api_key}'
    response = requests.get(url)
    data = response.json()
    status = data.get('status', None)
    if status != 'OK':
        raise ValueError(f'API Error: {data}')
    return data.get('results', None)