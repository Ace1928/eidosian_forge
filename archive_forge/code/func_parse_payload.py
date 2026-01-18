import json
import os
import signal
from typing import TYPE_CHECKING, Any, Dict
from rq.exceptions import InvalidJobOperation
from rq.job import Job
def parse_payload(payload: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Returns a dict of command data

    Args:
        payload (dict): Parses the payload dict.
    """
    return json.loads(payload.get('data').decode())