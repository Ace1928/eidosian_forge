import re
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple
def judge_direction(self, relation: str) -> str:
    """
        Args:
            relation: relation in string format
        """
    direction = 'BIDIRECTIONAL'
    if relation[0] == '<':
        direction = 'INCOMING'
    if relation[-1] == '>':
        direction = 'OUTGOING'
    return direction