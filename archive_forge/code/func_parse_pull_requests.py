from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def parse_pull_requests(self, pull_requests: List[PullRequest]) -> List[dict]:
    """
        Extracts title and number from each Issue and puts them in a dictionary
        Parameters:
            issues(List[Issue]): A list of Github Issue objects
        Returns:
            List[dict]: A dictionary of issue titles and numbers
        """
    parsed = []
    for pr in pull_requests:
        parsed.append({'title': pr.title, 'number': pr.number, 'commits': str(pr.commits), 'comments': str(pr.comments)})
    return parsed