import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
def execute_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
    """Handle an execute reply."""
    content = msg['content']
    user_expressions = content.setdefault('user_expressions', {})
    user_variables = content.pop('user_variables', {})
    if user_variables:
        user_expressions.update(user_variables)
    for payload in content.get('payload', []):
        if payload.get('source', None) == 'page' and 'text' in payload:
            if 'data' not in payload:
                payload['data'] = {}
            payload['data']['text/plain'] = payload.pop('text')
    return msg