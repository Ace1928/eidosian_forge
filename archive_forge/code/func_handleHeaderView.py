import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def handleHeaderView(self, elem, name, header):
    value = self.wprops.getAttribute(elem, name + 'Visible')
    if value is not None:
        header.setVisible(value)
    value = self.wprops.getAttribute(elem, name + 'CascadingSectionResizes')
    if value is not None:
        header.setCascadingSectionResizes(value)
    value = self.wprops.getAttribute(elem, name + 'DefaultSectionSize')
    if value is not None:
        header.setDefaultSectionSize(value)
    value = self.wprops.getAttribute(elem, name + 'HighlightSections')
    if value is not None:
        header.setHighlightSections(value)
    value = self.wprops.getAttribute(elem, name + 'MinimumSectionSize')
    if value is not None:
        header.setMinimumSectionSize(value)
    value = self.wprops.getAttribute(elem, name + 'ShowSortIndicator')
    if value is not None:
        header.setSortIndicatorShown(value)
    value = self.wprops.getAttribute(elem, name + 'StretchLastSection')
    if value is not None:
        header.setStretchLastSection(value)