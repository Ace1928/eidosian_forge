import collections
import contextlib
import dataclasses
import os
import shutil
import tempfile
import textwrap
import time
from typing import cast, Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple
import uuid
import torch
def setup_str(self) -> str:
    return '' if self.setup == 'pass' or not self.setup else f'setup:\n{textwrap.indent(self.setup, '  ')}' if '\n' in self.setup else f'setup: {self.setup}'