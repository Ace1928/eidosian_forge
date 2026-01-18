import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
def inspect_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
    """inspect_reply can't be easily backward compatible"""
    content = msg['content']
    new_content = msg['content'] = {'status': 'ok'}
    found = new_content['found'] = content['found']
    new_content['data'] = data = {}
    new_content['metadata'] = {}
    if found:
        lines = []
        for key in ('call_def', 'init_definition', 'definition'):
            if content.get(key, False):
                lines.append(content[key])
                break
        for key in ('call_docstring', 'init_docstring', 'docstring'):
            if content.get(key, False):
                lines.append(content[key])
                break
        if not lines:
            lines.append('<empty docstring>')
        data['text/plain'] = '\n'.join(lines)
    return msg