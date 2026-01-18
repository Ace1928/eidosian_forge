from math import pi, sin
from struct import pack
from PySide2.QtCore import QByteArray, QIODevice, Qt, QTimer, qWarning
from PySide2.QtMultimedia import (QAudio, QAudioDeviceInfo, QAudioFormat,
from PySide2.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
def createAudioOutput(self):
    self.m_audioOutput = QAudioOutput(self.m_device, self.m_format)
    self.m_audioOutput.notify.connect(self.notified)
    self.m_audioOutput.stateChanged.connect(self.handleStateChanged)
    self.m_generator.start()
    self.m_audioOutput.start(self.m_generator)
    self.m_volumeSlider.setValue(self.m_audioOutput.volume() * 100)