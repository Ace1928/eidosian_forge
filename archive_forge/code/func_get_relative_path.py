import sys
from pathlib import Path
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel
def get_relative_path(self, file_path: str) -> Path:
    """Get the relative path, returning an error if unsupported."""
    if self.root_dir is None:
        return Path(file_path)
    return get_validated_relative_path(Path(self.root_dir), file_path)