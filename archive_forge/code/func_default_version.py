import re
from typing import Optional, List, Union, Optional, Any, Dict
from dataclasses import dataclass
from lazyops import get_logger, PathIO
from lazyops.lazyclasses import lazyclass
@property
def default_version(self):
    if self.model_versions:
        for ver in self.model_versions:
            if ver.label is not None:
                return str(ver.step)
    return None