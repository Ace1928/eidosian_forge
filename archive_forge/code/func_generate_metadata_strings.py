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
def generate_metadata_strings(resources, metatype):

    def nothing_or_string(v):
        return '"{}"'.format(v) if v else 'nothing'
    return [jl_resource_tuple_string.format(relative_package_path=nothing_or_string(resource.get('relative_package_path', '')), external_url=nothing_or_string(resource.get('external_url', '')), dynamic=str(resource.get('dynamic', 'nothing')).lower(), type=metatype, async_string=':{}'.format(str(resource.get('async')).lower()) if 'async' in resource.keys() else 'nothing') for resource in resources]