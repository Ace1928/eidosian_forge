import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
def op_name_from_group(g: NativeFunctionsGroup) -> str:
    return g.functional.func.name.name.base