import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def flavorFor(self, mime):
    if mime == 'application/x-mycompany-VCard':
        return 'public.vcard'
    else:
        return ''