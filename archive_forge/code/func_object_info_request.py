import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
def object_info_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
    """Handle an object info request."""
    content = msg['content']
    code = content['code']
    cursor_pos = content['cursor_pos']
    line, _ = code_to_line(code, cursor_pos)
    new_content = msg['content'] = {}
    new_content['oname'] = extract_oname_v4(code, cursor_pos)
    new_content['detail_level'] = content['detail_level']
    return msg