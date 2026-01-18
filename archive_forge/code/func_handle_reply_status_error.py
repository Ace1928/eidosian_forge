import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
def handle_reply_status_error(self, msg: Dict[str, Any]) -> Dict[str, Any]:
    """This will be called *instead of* the regular handler

        on any reply with status != ok
        """
    return msg