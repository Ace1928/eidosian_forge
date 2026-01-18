import re
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple
def detect_labels(self, str_node: str, node_variable_dict: Dict[str, Any]) -> List[str]:
    """
        Args:
            str_node: node in string format
            node_variable_dict: dictionary of node variables
        """
    splitted_node = str_node.split(':')
    variable = splitted_node[0]
    labels = []
    if variable in node_variable_dict:
        labels = node_variable_dict[variable]
    elif variable == '' and len(splitted_node) > 1:
        labels = splitted_node[1:]
    return labels