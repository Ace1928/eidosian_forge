import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def attempt_parse_teams(self, input_dict: dict) -> Dict[str, List[dict]]:
    """Parse appropriate content from the list of teams."""
    parsed_teams: Dict[str, List[dict]] = {'teams': []}
    for team in input_dict['teams']:
        try:
            team = parse_dict_through_component(team, Team, fault_tolerant=False)
            parsed_teams['teams'].append(team)
        except Exception as e:
            warnings.warn(f'Error parsing a team {e}')
    return parsed_teams