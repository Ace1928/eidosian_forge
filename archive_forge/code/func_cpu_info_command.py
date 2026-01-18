import platform
from dataclasses import field
from enum import Enum
from typing import Dict, List, Optional, Union
from . import is_pydantic_available
from .doc import generate_doc_dataclass
def cpu_info_command():
    if platform.system() == 'Linux':
        return 'lscpu'
    elif platform.system() == 'Darwin':
        return 'sysctl -a | grep machdep.cpu'
    else:
        raise NotImplementedError('OS not supported.')