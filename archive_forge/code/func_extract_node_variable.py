import re
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple
def extract_node_variable(self, part: str) -> Optional[str]:
    """
        Args:
            part: node in string format
        """
    part = part.lstrip('(').rstrip(')')
    idx = part.find(':')
    if idx != -1:
        part = part[:idx]
    return None if part == '' else part