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
def readGlyphElement(self, glyphElement, instanceObject):
    """
        Read the glyph element, which could look like either one of these:

        .. code-block:: xml

            <glyph name="b" unicode="0x62"/>

            <glyph name="b"/>

            <glyph name="b">
                <master location="location-token-bbb" source="master-token-aaa2"/>
                <master glyphname="b.alt1" location="location-token-ccc" source="master-token-aaa3"/>
                <note>
                    This is an instance from an anisotropic interpolation.
                </note>
            </glyph>
        """
    glyphData = {}
    glyphName = glyphElement.attrib.get('name')
    if glyphName is None:
        raise DesignSpaceDocumentError('Glyph object without name attribute')
    mute = glyphElement.attrib.get('mute')
    if mute == '1':
        glyphData['mute'] = True
    unicodes = glyphElement.attrib.get('unicode')
    if unicodes is not None:
        try:
            unicodes = [int(u, 16) for u in unicodes.split(' ')]
            glyphData['unicodes'] = unicodes
        except ValueError:
            raise DesignSpaceDocumentError('unicode values %s are not integers' % unicodes)
    for noteElement in glyphElement.findall('.note'):
        glyphData['note'] = noteElement.text
        break
    designLocation, userLocation = self.locationFromElement(glyphElement)
    if userLocation:
        raise DesignSpaceDocumentError(f'<glyph> element "{glyphName}" must only have design locations (using xvalue="").')
    if designLocation is not None:
        glyphData['instanceLocation'] = designLocation
    glyphSources = None
    for masterElement in glyphElement.findall('.masters/master'):
        fontSourceName = masterElement.attrib.get('source')
        designLocation, userLocation = self.locationFromElement(masterElement)
        if userLocation:
            raise DesignSpaceDocumentError(f'<master> element "{fontSourceName}" must only have design locations (using xvalue="").')
        masterGlyphName = masterElement.attrib.get('glyphname')
        if masterGlyphName is None:
            masterGlyphName = glyphName
        d = dict(font=fontSourceName, location=designLocation, glyphName=masterGlyphName)
        if glyphSources is None:
            glyphSources = []
        glyphSources.append(d)
    if glyphSources is not None:
        glyphData['masters'] = glyphSources
    instanceObject.glyphs[glyphName] = glyphData