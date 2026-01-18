import json
from typing import Any, Dict
from langchain_core.load.serializable import Serializable, to_json_not_implemented
def dumpd(obj: Any) -> Dict[str, Any]:
    """Return a json dict representation of an object."""
    return json.loads(dumps(obj))