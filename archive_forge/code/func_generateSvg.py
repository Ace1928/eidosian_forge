import contextlib
import re
import xml.dom.minidom as xml
import numpy as np
from .. import debug
from .. import functions as fn
from ..parametertree import Parameter
from ..Qt import QtCore, QtGui, QtSvg, QtWidgets
from .Exporter import Exporter
def generateSvg(item, options=None):
    if options is None:
        options = {}
    global xmlHeader
    try:
        node, defs = _generateItemSvg(item, options=options)
    finally:
        if isinstance(item, QtWidgets.QGraphicsScene):
            items = item.items()
        else:
            items = [item]
            for i in items:
                items.extend(i.childItems())
        for i in items:
            if hasattr(i, 'setExportMode'):
                i.setExportMode(False)
    cleanXml(node)
    defsXml = '<defs>\n'
    for d in defs:
        defsXml += d.toprettyxml(indent='    ')
    defsXml += '</defs>\n'
    svgAttributes = f' viewBox ="0 0 {int(options['width'])} {int(options['height'])}"'
    c = options['background']
    backgroundtag = f'<rect width="100%" height="100%" fill="{c.name()}" fill-opacity="{c.alphaF()}" />\n'
    return xmlHeader % svgAttributes + backgroundtag + defsXml + node.toprettyxml(indent='    ') + '\n</svg>\n'