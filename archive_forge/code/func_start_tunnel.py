import atexit
import os
import platform
import re
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import List
import httpx
def start_tunnel(self) -> str:
    self.download_binary()
    self.url = self._start_tunnel(BINARY_PATH)
    return self.url