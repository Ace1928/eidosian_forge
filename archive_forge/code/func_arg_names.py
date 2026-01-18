import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
@property
def arg_names(self) -> List[str]:
    return self._arg_names