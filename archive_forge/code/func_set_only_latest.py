import re
from typing import Optional, List, Union, Optional, Any, Dict
from dataclasses import dataclass
from lazyops import get_logger, PathIO
from lazyops.lazyclasses import lazyclass
def set_only_latest(self):
    if self.model_versions:
        self.model_versions = [version for version in self.model_versions if version.label == 'latest']