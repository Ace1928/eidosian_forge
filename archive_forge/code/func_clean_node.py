import re
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple
def clean_node(self, node: str) -> str:
    """
        Args:
            node: node in string format

        """
    node = re.sub(self.property_pattern, '', node)
    node = node.replace('(', '')
    node = node.replace(')', '')
    node = node.strip()
    return node