from PySide2.QtWidgets import *
from PySide2.QtCore import *
def onTransition(self, e):
    self.p = PongEvent()
    machine.postDelayedEvent(self.p, 500)
    print('pong!')