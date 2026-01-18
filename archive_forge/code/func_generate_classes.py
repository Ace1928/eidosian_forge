import collections
import json
import os
from ._py_components_generation import (
from .base_component import ComponentRegistry
def generate_classes(namespace, metadata_path='lib/metadata.json'):
    """Load React component metadata into a format Dash can parse, then create
    Python class files.

    Usage: generate_classes()

    Keyword arguments:
    namespace -- name of the generated Python package (also output dir)

    metadata_path -- a path to a JSON file created by
    [`react-docgen`](https://github.com/reactjs/react-docgen).

    Returns:
    """
    data = _get_metadata(metadata_path)
    imports_path = os.path.join(namespace, '_imports_.py')
    if os.path.exists(imports_path):
        os.remove(imports_path)
    components = generate_classes_files(namespace, data, generate_class_file)
    generate_imports(namespace, components)