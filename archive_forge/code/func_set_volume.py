import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
def set_volume(self, volume: float) -> None:
    """Set the volume of the sound."""
    self.volume = volume