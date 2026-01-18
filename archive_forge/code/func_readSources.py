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
def readSources(self):
    for sourceCount, sourceElement in enumerate(self.root.findall('.sources/source')):
        filename = sourceElement.attrib.get('filename')
        if filename is not None and self.path is not None:
            sourcePath = os.path.abspath(os.path.join(os.path.dirname(self.path), filename))
        else:
            sourcePath = None
        sourceName = sourceElement.attrib.get('name')
        if sourceName is None:
            sourceName = 'temp_master.%d' % sourceCount
        sourceObject = self.sourceDescriptorClass()
        sourceObject.path = sourcePath
        sourceObject.filename = filename
        sourceObject.name = sourceName
        familyName = sourceElement.attrib.get('familyname')
        if familyName is not None:
            sourceObject.familyName = familyName
        styleName = sourceElement.attrib.get('stylename')
        if styleName is not None:
            sourceObject.styleName = styleName
        for familyNameElement in sourceElement.findall('familyname'):
            for key, lang in familyNameElement.items():
                if key == XML_LANG:
                    familyName = familyNameElement.text
                    sourceObject.setFamilyName(familyName, lang)
        designLocation, userLocation = self.locationFromElement(sourceElement)
        if userLocation:
            raise DesignSpaceDocumentError(f'<source> element "{sourceName}" must only have design locations (using xvalue="").')
        sourceObject.location = designLocation
        layerName = sourceElement.attrib.get('layer')
        if layerName is not None:
            sourceObject.layerName = layerName
        for libElement in sourceElement.findall('.lib'):
            if libElement.attrib.get('copy') == '1':
                sourceObject.copyLib = True
        for groupsElement in sourceElement.findall('.groups'):
            if groupsElement.attrib.get('copy') == '1':
                sourceObject.copyGroups = True
        for infoElement in sourceElement.findall('.info'):
            if infoElement.attrib.get('copy') == '1':
                sourceObject.copyInfo = True
            if infoElement.attrib.get('mute') == '1':
                sourceObject.muteInfo = True
        for featuresElement in sourceElement.findall('.features'):
            if featuresElement.attrib.get('copy') == '1':
                sourceObject.copyFeatures = True
        for glyphElement in sourceElement.findall('.glyph'):
            glyphName = glyphElement.attrib.get('name')
            if glyphName is None:
                continue
            if glyphElement.attrib.get('mute') == '1':
                sourceObject.mutedGlyphNames.append(glyphName)
        for kerningElement in sourceElement.findall('.kerning'):
            if kerningElement.attrib.get('mute') == '1':
                sourceObject.muteKerning = True
        self.documentObject.sources.append(sourceObject)