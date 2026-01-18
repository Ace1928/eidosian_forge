import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
def kernel_info_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
    """Handle a kernel info reply."""
    content = msg['content']
    for key in ('protocol_version', 'ipython_version'):
        if key in content:
            content[key] = '.'.join(map(str, content[key]))
    content.setdefault('protocol_version', '4.1')
    if content['language'].startswith('python') and 'ipython_version' in content:
        content['implementation'] = 'ipython'
        content['implementation_version'] = content.pop('ipython_version')
    language = content.pop('language')
    language_info = content.setdefault('language_info', {})
    language_info.setdefault('name', language)
    if 'language_version' in content:
        language_version = '.'.join(map(str, content.pop('language_version')))
        language_info.setdefault('version', language_version)
    content['banner'] = ''
    return msg