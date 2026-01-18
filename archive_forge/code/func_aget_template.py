from __future__ import annotations
import abc
from pathlib import Path
from typing import Optional, Any, Dict, List, Union, Type, TYPE_CHECKING
def aget_template(self, *parts: str) -> 'jinja2.Template':
    """
        Gets the template
        """
    if not parts[-1].endswith(self.file_suffix):
        parts = list(parts[:-1]) + [parts[-1] + self.file_suffix]
    return self.aj2.get_template('/'.join(parts))