import re
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple
def correct_query(self, query: str) -> str:
    """
        Args:
            query: cypher query
        """
    node_variable_dict = self.detect_node_variables(query)
    paths = self.extract_paths(query)
    for path in paths:
        original_path = path
        start_idx = 0
        while start_idx < len(path):
            match_res = re.match(self.node_relation_node_pattern, path[start_idx:])
            if match_res is None:
                break
            start_idx += match_res.start()
            match_dict = match_res.groupdict()
            left_node_labels = self.detect_labels(match_dict['left_node'], node_variable_dict)
            right_node_labels = self.detect_labels(match_dict['right_node'], node_variable_dict)
            end_idx = start_idx + 4 + len(match_dict['left_node']) + len(match_dict['relation']) + len(match_dict['right_node'])
            original_partial_path = original_path[start_idx:end_idx + 1]
            relation_direction, relation_types = self.detect_relation_types(match_dict['relation'])
            if relation_types != [] and ''.join(relation_types).find('*') != -1:
                start_idx += len(match_dict['left_node']) + len(match_dict['relation']) + 2
                continue
            if relation_direction == 'OUTGOING':
                is_legal = self.verify_schema(left_node_labels, relation_types, right_node_labels)
                if not is_legal:
                    is_legal = self.verify_schema(right_node_labels, relation_types, left_node_labels)
                    if is_legal:
                        corrected_relation = '<' + match_dict['relation'][:-1]
                        corrected_partial_path = original_partial_path.replace(match_dict['relation'], corrected_relation)
                        query = query.replace(original_partial_path, corrected_partial_path)
                    else:
                        return ''
            elif relation_direction == 'INCOMING':
                is_legal = self.verify_schema(right_node_labels, relation_types, left_node_labels)
                if not is_legal:
                    is_legal = self.verify_schema(left_node_labels, relation_types, right_node_labels)
                    if is_legal:
                        corrected_relation = match_dict['relation'][1:] + '>'
                        corrected_partial_path = original_partial_path.replace(match_dict['relation'], corrected_relation)
                        query = query.replace(original_partial_path, corrected_partial_path)
                    else:
                        return ''
            else:
                is_legal = self.verify_schema(left_node_labels, relation_types, right_node_labels)
                is_legal |= self.verify_schema(right_node_labels, relation_types, left_node_labels)
                if not is_legal:
                    return ''
            start_idx += len(match_dict['left_node']) + len(match_dict['relation']) + 2
    return query