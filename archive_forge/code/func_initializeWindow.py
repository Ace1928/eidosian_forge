from math import pi, sin
from struct import pack
from PySide2.QtCore import QByteArray, QIODevice, Qt, QTimer, qWarning
from PySide2.QtMultimedia import (QAudio, QAudioDeviceInfo, QAudioFormat,
from PySide2.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
def initializeWindow(self):
    layout = QVBoxLayout()
    self.m_deviceBox = QComboBox()
    self.m_deviceBox.activated[int].connect(self.deviceChanged)
    for deviceInfo in QAudioDeviceInfo.availableDevices(QAudio.AudioOutput):
        self.m_deviceBox.addItem(deviceInfo.deviceName(), deviceInfo)
    layout.addWidget(self.m_deviceBox)
    self.m_modeButton = QPushButton()
    self.m_modeButton.clicked.connect(self.toggleMode)
    self.m_modeButton.setText(self.PUSH_MODE_LABEL)
    layout.addWidget(self.m_modeButton)
    self.m_suspendResumeButton = QPushButton(clicked=self.toggleSuspendResume)
    self.m_suspendResumeButton.setText(self.SUSPEND_LABEL)
    layout.addWidget(self.m_suspendResumeButton)
    volumeBox = QHBoxLayout()
    volumeLabel = QLabel('Volume:')
    self.m_volumeSlider = QSlider(Qt.Horizontal, minimum=0, maximum=100, singleStep=10)
    self.m_volumeSlider.valueChanged.connect(self.volumeChanged)
    volumeBox.addWidget(volumeLabel)
    volumeBox.addWidget(self.m_volumeSlider)
    layout.addLayout(volumeBox)
    window = QWidget()
    window.setLayout(layout)
    self.setCentralWidget(window)