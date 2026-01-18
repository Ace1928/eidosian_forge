from math import pi, sin
from struct import pack
from PySide2.QtCore import QByteArray, QIODevice, Qt, QTimer, qWarning
from PySide2.QtMultimedia import (QAudio, QAudioDeviceInfo, QAudioFormat,
from PySide2.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
def generateData(self, format, durationUs, sampleRate):
    pack_format = ''
    if format.sampleSize() == 8:
        if format.sampleType() == QAudioFormat.UnSignedInt:
            scaler = lambda x: (1.0 + x) / 2 * 255
            pack_format = 'B'
        elif format.sampleType() == QAudioFormat.SignedInt:
            scaler = lambda x: x * 127
            pack_format = 'b'
    elif format.sampleSize() == 16:
        if format.sampleType() == QAudioFormat.UnSignedInt:
            scaler = lambda x: (1.0 + x) / 2 * 65535
            pack_format = '<H' if format.byteOrder() == QAudioFormat.LittleEndian else '>H'
        elif format.sampleType() == QAudioFormat.SignedInt:
            scaler = lambda x: x * 32767
            pack_format = '<h' if format.byteOrder() == QAudioFormat.LittleEndian else '>h'
    assert pack_format != ''
    channelBytes = format.sampleSize() // 8
    sampleBytes = format.channelCount() * channelBytes
    length = format.sampleRate() * format.channelCount() * (format.sampleSize() // 8) * durationUs // 100000
    self.m_buffer.clear()
    sampleIndex = 0
    factor = 2 * pi * sampleRate / format.sampleRate()
    while length != 0:
        x = sin(sampleIndex % format.sampleRate() * factor)
        packed = pack(pack_format, int(scaler(x)))
        for _ in range(format.channelCount()):
            self.m_buffer.append(packed)
            length -= channelBytes
        sampleIndex += 1