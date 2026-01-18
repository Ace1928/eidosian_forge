import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
@property
def ll_file(self) -> str:
    return self._ll_file