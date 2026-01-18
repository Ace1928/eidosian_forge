import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def setupObject(self, clsname, parent, branch, is_attribute=True):
    name = self.uniqueName(branch.attrib.get('name') or clsname[1:].lower())
    if parent is None:
        args = ()
    else:
        args = (parent,)
    obj = self.factory.createQObject(clsname, name, args, is_attribute)
    self.wprops.setProperties(obj, branch)
    obj.setObjectName(name)
    if is_attribute:
        setattr(self.toplevelWidget, name, obj)
    return obj