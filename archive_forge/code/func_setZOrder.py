import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def setZOrder(self, elem):
    if elem.text is None:
        return
    try:
        getattr(self.toplevelWidget, elem.text).raise_()
    except AttributeError:
        pass