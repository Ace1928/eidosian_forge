from math import pi, sin
from struct import pack
from PySide2.QtCore import QByteArray, QIODevice, Qt, QTimer, qWarning
from PySide2.QtMultimedia import (QAudio, QAudioDeviceInfo, QAudioFormat,
from PySide2.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
def handleStateChanged(self, state):
    qWarning('state = ' + self.stateMap.get(state, 'Unknown'))