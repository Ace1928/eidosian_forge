import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union
def parse_datapipe_files(file_paths: Set[str]) -> Tuple[Dict[str, str], Dict[str, str], Set[str], Dict[str, List[str]]]:
    methods_and_signatures, methods_and_class_names, methods_with_special_output_types = ({}, {}, set())
    methods_and_doc_strings = {}
    for path in file_paths:
        method_to_signature, method_to_class_name, methods_needing_special_output_types, doc_string_dict = parse_datapipe_file(path)
        methods_and_signatures.update(method_to_signature)
        methods_and_class_names.update(method_to_class_name)
        methods_with_special_output_types.update(methods_needing_special_output_types)
        methods_and_doc_strings.update(doc_string_dict)
    return (methods_and_signatures, methods_and_class_names, methods_with_special_output_types, methods_and_doc_strings)