import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
def complete_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
    """Handle a complete reply."""
    content = msg['content']
    new_content = msg['content'] = {'status': 'ok'}
    new_content['matches'] = content['matches']
    if content['matched_text']:
        new_content['cursor_start'] = -len(content['matched_text'])
    else:
        new_content['cursor_start'] = None
    new_content['cursor_end'] = None
    new_content['metadata'] = {}
    return msg