import logging
from typing import Any, Dict, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def fetch_place_details(self, place_id: str) -> Optional[str]:
    try:
        place_details = self.google_map_client.place(place_id)
        place_details['place_id'] = place_id
        formatted_details = self.format_place_details(place_details)
        return formatted_details
    except Exception as e:
        logging.error(f'An Error occurred while fetching place details: {e}')
        return None