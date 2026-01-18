from math import pi, sin
from struct import pack
from PySide2.QtCore import QByteArray, QIODevice, Qt, QTimer, qWarning
from PySide2.QtMultimedia import (QAudio, QAudioDeviceInfo, QAudioFormat,
from PySide2.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
def pullTimerExpired(self):
    if self.m_audioOutput is not None and self.m_audioOutput.state() != QAudio.StoppedState:
        chunks = self.m_audioOutput.bytesFree() // self.m_audioOutput.periodSize()
        for _ in range(chunks):
            data = self.m_generator.read(self.m_audioOutput.periodSize())
            if data is None or len(data) != self.m_audioOutput.periodSize():
                break
            self.m_output.write(data)