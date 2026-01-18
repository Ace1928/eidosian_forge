from ..Qt import QtCore, QtGui, QtWidgets
def nameChanged(self, param, name):
    if self.param.opts.get('title', None) is None:
        self.titleChanged()