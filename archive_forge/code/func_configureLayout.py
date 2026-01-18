import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def configureLayout(self, elem, layout):
    if isinstance(layout, QtWidgets.QGridLayout):
        self.setArray(elem, 'columnminimumwidth', layout.setColumnMinimumWidth)
        self.setArray(elem, 'rowminimumheight', layout.setRowMinimumHeight)
        self.setArray(elem, 'columnstretch', layout.setColumnStretch)
        self.setArray(elem, 'rowstretch', layout.setRowStretch)
    elif isinstance(layout, QtWidgets.QBoxLayout):
        self.setArray(elem, 'stretch', layout.setStretch)