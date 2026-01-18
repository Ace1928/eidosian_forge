from ..Qt import QtWidgets
def nextColumn(self, colspan=1):
    """Advance to next column, while returning the current column number 
        (generally only for internal use--called by addWidget)"""
    self.currentCol += colspan
    return self.currentCol - colspan