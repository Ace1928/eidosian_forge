from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from uuid import UUID, uuid4
import urllib.parse
from langsmith import schemas as ls_schemas
from langsmith import utils
from langsmith.client import ID_TYPE, RUN_TYPE_T, Client, _dumps_json
@root_validator(pre=False)
def ensure_dotted_order(cls, values: dict) -> dict:
    """Ensure the dotted order of the run."""
    current_dotted_order = values.get('dotted_order')
    if current_dotted_order and current_dotted_order.strip():
        return values
    current_dotted_order = _create_current_dotted_order(values['start_time'], values['id'])
    if values['parent_run']:
        values['dotted_order'] = values['parent_run'].dotted_order + '.' + current_dotted_order
    else:
        values['dotted_order'] = current_dotted_order
    return values