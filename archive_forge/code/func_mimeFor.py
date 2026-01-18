import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def mimeFor(self, flav):
    if flav == 'public.vcard':
        return 'application/x-mycompany-VCard'
    else:
        return ''