import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
def input_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
    """Handle an input request."""
    msg['content'].setdefault('password', False)
    return msg