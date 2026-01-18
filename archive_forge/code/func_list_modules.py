import sys
from PyQt5 import QtWidgets
from DAWModules import SoundModule
from PyQt5.QtCore import Qt
import pyaudio
import numpy as np
from typing import Dict, Type, List, Tuple
def list_modules(self) -> List[str]:
    """
        Lists all registered modules' names.

        Returns:
            List[str]: A list of all registered module names.
        """
    return list(self.modules.keys())