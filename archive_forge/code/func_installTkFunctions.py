import tkinter.messagebox as tkMessageBox
import tkinter.simpledialog as tkSimpleDialog
from twisted.internet import task
def installTkFunctions():
    import twisted.python.util
    twisted.python.util.getPassword = getPassword