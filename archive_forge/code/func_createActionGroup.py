import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def createActionGroup(self, elem):
    action_group = self.setupObject('QActionGroup', self.toplevelWidget, elem)
    self.currentActionGroup = action_group
    self.traverseWidgetTree(elem)
    self.currentActionGroup = None