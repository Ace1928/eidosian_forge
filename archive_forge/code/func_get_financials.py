import json
from typing import Any, Dict, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
def get_financials(self, ticker: str) -> Optional[dict]:
    """
        Get fundamental financial data, which is found in balance sheets,
        income statements, and cash flow statements for a given ticker.

        /vX/reference/financials
        """
    url = f'{POLYGON_BASE_URL}vX/reference/financials?ticker={ticker}&apiKey={self.polygon_api_key}'
    response = requests.get(url)
    data = response.json()
    status = data.get('status', None)
    if status != 'OK':
        raise ValueError(f'API Error: {data}')
    return data.get('results', None)