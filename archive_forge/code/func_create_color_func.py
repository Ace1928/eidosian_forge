import os
import re
import shutil
import sys
from typing import Dict, Pattern
def create_color_func(name: str) -> None:

    def inner(text: str) -> str:
        return colorize(name, text)
    globals()[name] = inner