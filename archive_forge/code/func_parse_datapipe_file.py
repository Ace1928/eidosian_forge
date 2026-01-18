import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union
def parse_datapipe_file(file_path: str) -> Tuple[Dict[str, str], Dict[str, str], Set[str], Dict[str, List[str]]]:
    """Given a path to file, parses the file and returns a dictionary of method names to function signatures."""
    method_to_signature, method_to_class_name, special_output_type = ({}, {}, set())
    doc_string_dict = defaultdict(list)
    with open(file_path) as f:
        open_paren_count = 0
        method_name, class_name, signature = ('', '', '')
        skip = False
        for line in f.readlines():
            if line.count('"""') % 2 == 1:
                skip = not skip
            if skip or '"""' in line:
                doc_string_dict[method_name].append(line)
                continue
            if '@functional_datapipe' in line:
                method_name = extract_method_name(line)
                doc_string_dict[method_name] = []
                continue
            if method_name and 'class ' in line:
                class_name = extract_class_name(line)
                continue
            if method_name and ('def __init__(' in line or 'def __new__(' in line):
                if 'def __new__(' in line:
                    special_output_type.add(method_name)
                open_paren_count += 1
                start = line.find('(') + len('(')
                line = line[start:]
            if open_paren_count > 0:
                open_paren_count += line.count('(')
                open_paren_count -= line.count(')')
                if open_paren_count == 0:
                    end = line.rfind(')')
                    signature += line[:end]
                    method_to_signature[method_name] = process_signature(signature)
                    method_to_class_name[method_name] = class_name
                    method_name, class_name, signature = ('', '', '')
                elif open_paren_count < 0:
                    raise RuntimeError("open parenthesis count < 0. This shouldn't be possible.")
                else:
                    signature += line.strip('\n').strip(' ')
    return (method_to_signature, method_to_class_name, special_output_type, doc_string_dict)