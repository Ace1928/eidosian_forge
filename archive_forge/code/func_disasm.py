import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
def disasm(self, lib_path: str) -> None:
    subprocess.Popen([self._path, lib_path, '-o', self.ll_file], stdout=subprocess.PIPE).communicate()