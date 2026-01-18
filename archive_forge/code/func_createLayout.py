import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def createLayout(self, elem):
    margin = -1 if self.stack.topIsLayout() else self.defaults['margin']
    margin = self.wprops.getProperty(elem, 'margin', margin)
    left = self.wprops.getProperty(elem, 'leftMargin', margin)
    top = self.wprops.getProperty(elem, 'topMargin', margin)
    right = self.wprops.getProperty(elem, 'rightMargin', margin)
    bottom = self.wprops.getProperty(elem, 'bottomMargin', margin)
    if self.stack.topIsLayoutWidget():
        if left < 0:
            left = 0
        if top < 0:
            top = 0
        if right < 0:
            right = 0
        if bottom < 0:
            bottom = 0
    if left >= 0 or top >= 0 or right >= 0 or (bottom >= 0):
        cme = SubElement(elem, 'property', name='pyuicMargins')
        SubElement(cme, 'number').text = str(left)
        SubElement(cme, 'number').text = str(top)
        SubElement(cme, 'number').text = str(right)
        SubElement(cme, 'number').text = str(bottom)
    spacing = self.wprops.getProperty(elem, 'spacing', self.defaults['spacing'])
    horiz = self.wprops.getProperty(elem, 'horizontalSpacing', spacing)
    vert = self.wprops.getProperty(elem, 'verticalSpacing', spacing)
    if horiz >= 0 or vert >= 0:
        cme = SubElement(elem, 'property', name='pyuicSpacing')
        SubElement(cme, 'number').text = str(horiz)
        SubElement(cme, 'number').text = str(vert)
    classname = elem.attrib['class']
    if self.stack.topIsLayout():
        parent = None
    else:
        parent = self.stack.topwidget
    if 'name' not in elem.attrib:
        elem.attrib['name'] = classname[1:].lower()
    self.stack.push(self.setupObject(classname, parent, elem))
    self.traverseWidgetTree(elem)
    layout = self.stack.popLayout()
    self.configureLayout(elem, layout)
    if self.stack.topIsLayout():
        top_layout = self.stack.peek()
        lp = elem.attrib['layout-position']
        if isinstance(top_layout, QtWidgets.QFormLayout):
            top_layout.setLayout(lp[0], self._form_layout_role(lp), layout)
        else:
            top_layout.addLayout(layout, *lp)