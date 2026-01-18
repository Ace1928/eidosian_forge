import re
import shlex
from typing import List
from mlflow.utils.os import is_windows
def generate_feature_name_if_not_string(s):
    if isinstance(s, str):
        return s
    return f'feature_{s}'