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
def base_package_name(project_shortname):
    return 'DashBase' if is_core_package(project_shortname) else 'Dash'