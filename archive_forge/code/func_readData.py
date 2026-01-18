from math import pi, sin
from struct import pack
from PySide2.QtCore import QByteArray, QIODevice, Qt, QTimer, qWarning
from PySide2.QtMultimedia import (QAudio, QAudioDeviceInfo, QAudioFormat,
from PySide2.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
def readData(self, maxlen):
    data = QByteArray()
    total = 0
    while maxlen > total:
        chunk = min(self.m_buffer.size() - self.m_pos, maxlen - total)
        data.append(self.m_buffer.mid(self.m_pos, chunk))
        self.m_pos = (self.m_pos + chunk) % self.m_buffer.size()
        total += chunk
    return data.data()