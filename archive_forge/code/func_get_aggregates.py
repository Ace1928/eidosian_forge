import json
from typing import Any, Dict, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
def get_aggregates(self, ticker: str, **kwargs: Any) -> Optional[dict]:
    """
        Get aggregate bars for a stock over a given date range
        in custom time window sizes.

        /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}
        """
    timespan = kwargs.get('timespan', 'day')
    multiplier = kwargs.get('timespan_multiplier', 1)
    from_date = kwargs.get('from_date', None)
    to_date = kwargs.get('to_date', None)
    adjusted = kwargs.get('adjusted', True)
    sort = kwargs.get('sort', 'asc')
    url = f'{POLYGON_BASE_URL}v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}?apiKey={self.polygon_api_key}&adjusted={adjusted}&sort={sort}'
    response = requests.get(url)
    data = response.json()
    status = data.get('status', None)
    if status != 'OK':
        raise ValueError(f'API Error: {data}')
    return data.get('results', None)