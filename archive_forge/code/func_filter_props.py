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
def filter_props(props):
    """Filter props from the Component arguments to exclude:
        - Those without a "type" or a "flowType" field
        - Those with arg.type.name in {'func', 'symbol', 'instanceOf'}
    Parameters
    ----------
    props: dict
        Dictionary with {propName: propMetadata} structure
    Returns
    -------
    dict
        Filtered dictionary with {propName: propMetadata} structure
    """
    filtered_props = copy.deepcopy(props)
    for arg_name, arg in list(filtered_props.items()):
        if 'type' not in arg and 'flowType' not in arg:
            filtered_props.pop(arg_name)
            continue
        if 'type' in arg:
            arg_type = arg['type']['name']
            if arg_type in {'func', 'symbol', 'instanceOf'}:
                filtered_props.pop(arg_name)
        elif 'flowType' in arg:
            arg_type_name = arg['flowType']['name']
            if arg_type_name == 'signature':
                if 'type' not in arg['flowType'] or arg['flowType']['type'] != 'object':
                    filtered_props.pop(arg_name)
        else:
            raise ValueError
    return filtered_props