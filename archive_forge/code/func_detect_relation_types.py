import re
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple
def detect_relation_types(self, str_relation: str) -> Tuple[str, List[str]]:
    """
        Args:
            str_relation: relation in string format
        """
    relation_direction = self.judge_direction(str_relation)
    relation_type = self.relation_type_pattern.search(str_relation)
    if relation_type is None or relation_type.group('relation_type') is None:
        return (relation_direction, [])
    relation_types = [t.strip().strip('!') for t in relation_type.group('relation_type').split('|')]
    return (relation_direction, relation_types)