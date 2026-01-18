import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def createUserInterface(self, elem):
    cname = elem.attrib['class']
    wname = elem.attrib['name']
    if not wname:
        wname = cname
        if wname.startswith('Q'):
            wname = wname[1:]
        wname = wname[0].lower() + wname[1:]
    self.toplevelWidget = self.createToplevelWidget(cname, wname)
    self.toplevelWidget.setObjectName(wname)
    DEBUG('toplevel widget is %s', self.toplevelWidget.metaObject().className())
    self.wprops.setProperties(self.toplevelWidget, elem)
    self.stack.push(self.toplevelWidget)
    self.traverseWidgetTree(elem)
    self.stack.popWidget()
    self.addActions()
    self.setBuddies()
    self.setDelayedProps()