import logging
from typing import Any, Dict, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def format_place_details(self, place_details: Dict[str, Any]) -> Optional[str]:
    try:
        name = place_details.get('result', {}).get('name', 'Unknown')
        address = place_details.get('result', {}).get('formatted_address', 'Unknown')
        phone_number = place_details.get('result', {}).get('formatted_phone_number', 'Unknown')
        website = place_details.get('result', {}).get('website', 'Unknown')
        place_id = place_details.get('result', {}).get('place_id', 'Unknown')
        formatted_details = f'{name}\nAddress: {address}\nGoogle place ID: {place_id}\nPhone: {phone_number}\nWebsite: {website}\n\n'
        return formatted_details
    except Exception as e:
        logging.error(f'An error occurred while formatting place details: {e}')
        return None