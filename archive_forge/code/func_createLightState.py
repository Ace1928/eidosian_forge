from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
def createLightState(light, duration, parent=None):
    lightState = QState(parent)
    timer = QTimer(lightState)
    timer.setInterval(duration)
    timer.setSingleShot(True)
    timing = QState(lightState)
    timing.entered.connect(light.turnOn)
    timing.entered.connect(timer.start)
    timing.exited.connect(light.turnOff)
    done = QFinalState(lightState)
    timing.addTransition(timer, SIGNAL('timeout()'), done)
    lightState.setInitialState(timing)
    return lightState