import json
import re
import urllib.parse
from typing import Any, Dict, Iterable, Optional, Type, TypeVar, Union
def is_local_editable(self) -> bool:
    return isinstance(self.info, DirInfo) and self.info.editable