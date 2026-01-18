import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
@property
def ret_type(self) -> str:
    return self._ret_type