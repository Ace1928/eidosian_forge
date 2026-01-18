import json
import os
import subprocess
import sys
from typing import List, Tuple
from pathlib import Path
from . import (METATYPES_JSON_SUFFIX, PROJECT_FILE_SUFFIX, qt_metatype_json_dir,
def is_main(file):
    return '__main__' in file.read_text(encoding='utf-8')