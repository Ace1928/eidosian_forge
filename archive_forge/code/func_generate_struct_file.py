import copy
import os
import shutil
import warnings
import sys
import importlib
import uuid
import hashlib
from ._all_keywords import julia_keywords
from ._py_components_generation import reorder_props
def generate_struct_file(name, props, description, project_shortname, prefix):
    props = reorder_props(props=props)
    import_string = '# AUTO GENERATED FILE - DO NOT EDIT\n'
    class_string = generate_class_string(name, props, description, project_shortname, prefix)
    file_name = format_fn_name(prefix, name) + '.jl'
    if not os.path.exists('src/jl'):
        os.makedirs('src/jl')
    file_path = os.path.join('src', 'jl', file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(import_string)
        f.write(class_string)
    print('Generated {}'.format(file_name))