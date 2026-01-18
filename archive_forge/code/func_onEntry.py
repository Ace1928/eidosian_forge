from PySide2.QtWidgets import *
from PySide2.QtCore import *
def onEntry(self, e):
    self.p = PingEvent()
    self.machine().postEvent(self.p)
    print('ping?')