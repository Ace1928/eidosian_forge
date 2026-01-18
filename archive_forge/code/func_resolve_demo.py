from __future__ import annotations
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
import httpx
import semantic_version
from huggingface_hub import HfApi
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from tomlkit import parse
from typer import Argument, Option
from typing_extensions import Annotated
def resolve_demo(demo_dir: Path) -> Path:
    _demo_dir = demo_dir.resolve()
    if (_demo_dir / 'space.py').exists():
        return _demo_dir / 'space.py'
    elif (_demo_dir / 'app.py').exists():
        return _demo_dir / 'app.py'
    else:
        raise FileNotFoundError(f'Could not find "space.py" or "app.py" in "{demo_dir}".')