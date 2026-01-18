import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
def object_info_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
    """inspect_reply can't be easily backward compatible"""
    msg['content'] = {'found': False, 'oname': 'unknown'}
    return msg