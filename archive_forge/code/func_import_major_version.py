import json
import os
import subprocess
import sys
from typing import List, Tuple
from pathlib import Path
from . import (METATYPES_JSON_SUFFIX, PROJECT_FILE_SUFFIX, qt_metatype_json_dir,
@import_major_version.setter
def import_major_version(self, v):
    self._import_major_version = v