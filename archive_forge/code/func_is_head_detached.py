from __future__ import annotations
import os
import re
from typing import Any
from streamlit import util
@property
def is_head_detached(self):
    if not self.is_valid():
        return False
    return self.repo.head.is_detached