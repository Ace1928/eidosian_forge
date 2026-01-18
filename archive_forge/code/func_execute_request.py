import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
def execute_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
    """Handle an execute request."""
    content = msg['content']
    user_variables = content.pop('user_variables', [])
    user_expressions = content.setdefault('user_expressions', {})
    for v in user_variables:
        user_expressions[v] = v
    return msg