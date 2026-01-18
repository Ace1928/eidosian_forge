import os
def on_loaded(self):
    """Handle app load."""
    self.size = self.page().contentsSize().toSize()
    self.resize(self.size)
    QtCore.QTimer.singleShot(1000, self.export)