from PySide2.QtWidgets import (QMainWindow, QAction, QFileDialog, QApplication)
from addresswidget import AddressWidget
def createAction(self, text, menu, slot):
    """ Helper function to save typing when populating menus
            with action.
        """
    action = QAction(text, self)
    menu.addAction(action)
    action.triggered.connect(slot)
    return action