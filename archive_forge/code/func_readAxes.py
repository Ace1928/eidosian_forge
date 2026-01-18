from __future__ import annotations
import collections
import copy
import itertools
import math
import os
import posixpath
from io import BytesIO, StringIO
from textwrap import indent
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union, cast
from fontTools.misc import etree as ET
from fontTools.misc import plistlib
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr
def readAxes(self):
    axesElement = self.root.find('.axes')
    if axesElement is not None and 'elidedfallbackname' in axesElement.attrib:
        self.documentObject.elidedFallbackName = axesElement.attrib['elidedfallbackname']
    axisElements = self.root.findall('.axes/axis')
    if not axisElements:
        return
    for axisElement in axisElements:
        if self.documentObject.formatTuple >= (5, 0) and 'values' in axisElement.attrib:
            axisObject = self.discreteAxisDescriptorClass()
            axisObject.values = [float(s) for s in axisElement.attrib['values'].split(' ')]
        else:
            axisObject = self.axisDescriptorClass()
            axisObject.minimum = float(axisElement.attrib.get('minimum'))
            axisObject.maximum = float(axisElement.attrib.get('maximum'))
        axisObject.default = float(axisElement.attrib.get('default'))
        axisObject.name = axisElement.attrib.get('name')
        if axisElement.attrib.get('hidden', False):
            axisObject.hidden = True
        axisObject.tag = axisElement.attrib.get('tag')
        for mapElement in axisElement.findall('map'):
            a = float(mapElement.attrib['input'])
            b = float(mapElement.attrib['output'])
            axisObject.map.append((a, b))
        for labelNameElement in axisElement.findall('labelname'):
            for key, lang in labelNameElement.items():
                if key == XML_LANG:
                    axisObject.labelNames[lang] = tostr(labelNameElement.text)
        labelElement = axisElement.find('.labels')
        if labelElement is not None:
            if 'ordering' in labelElement.attrib:
                axisObject.axisOrdering = int(labelElement.attrib['ordering'])
            for label in labelElement.findall('.label'):
                axisObject.axisLabels.append(self.readAxisLabel(label))
        self.documentObject.axes.append(axisObject)
        self.axisDefaults[axisObject.name] = axisObject.default
    self.documentObject.axisMappings = []
    for mappingsElement in self.root.findall('.axes/mappings'):
        groupDescription = mappingsElement.attrib.get('description')
        for mappingElement in mappingsElement.findall('mapping'):
            description = mappingElement.attrib.get('description')
            inputElement = mappingElement.find('input')
            outputElement = mappingElement.find('output')
            inputLoc = {}
            outputLoc = {}
            for dimElement in inputElement.findall('.dimension'):
                name = dimElement.attrib['name']
                value = float(dimElement.attrib['xvalue'])
                inputLoc[name] = value
            for dimElement in outputElement.findall('.dimension'):
                name = dimElement.attrib['name']
                value = float(dimElement.attrib['xvalue'])
                outputLoc[name] = value
            axisMappingObject = self.axisMappingDescriptorClass(inputLocation=inputLoc, outputLocation=outputLoc, description=description, groupDescription=groupDescription)
            self.documentObject.axisMappings.append(axisMappingObject)