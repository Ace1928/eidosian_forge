import os
import shutil
import sys
from pathlib import Path
from subprocess import check_output
from typing import List, Text, Union
from ..schema import SPEC_VERSION
from ..types import (
def censored_spec(spec: LanguageServerSpec) -> LanguageServerSpec:
    return {k: SKIP_JSON_SPEC.get(k, v) for k, v in spec.items()}