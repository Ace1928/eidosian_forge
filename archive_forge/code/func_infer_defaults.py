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
@root_validator(pre=True)
def infer_defaults(cls, values: dict) -> dict:
    """Assign name to the run."""
    if 'serialized' not in values:
        values['serialized'] = {'name': values['name']}
    if values.get('parent_run') is not None:
        values['parent_run_id'] = values['parent_run'].id
    if 'id' not in values:
        values['id'] = uuid4()
    if 'trace_id' not in values:
        if 'parent_run' in values:
            values['trace_id'] = values['parent_run'].trace_id
        else:
            values['trace_id'] = values['id']
    cast(dict, values.setdefault('extra', {}))
    return values