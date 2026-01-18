import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
def is_valid_color(color_str: str) -> bool:
    hex_color_pattern = '^#(?:[0-9a-fA-F]{3}){1,2}$'
    if re.match(hex_color_pattern, color_str):
        return True
    try:
        if color_str.startswith('rgb(') and color_str.endswith(')'):
            parts = color_str[4:-1].split(',')
        elif color_str.startswith('rgba(') and color_str.endswith(')'):
            parts = color_str[5:-1].split(',')
        else:
            return False
        parts = [int(p.strip()) for p in parts]
        if len(parts) == 3 and all((0 <= p <= 255 for p in parts)):
            return True
        if len(parts) == 4 and all((0 <= p <= 255 for p in parts[:-1])) and (0 <= parts[-1] <= 1):
            return True
    except ValueError:
        pass
    return False