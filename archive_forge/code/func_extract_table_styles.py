import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def extract_table_styles(self, styles_str):
    root = etree.fromstring(styles_str)
    table_styles = {}
    auto_styles = root.find('{%s}automatic-styles' % (CNSD['office'],))
    for stylenode in auto_styles:
        name = stylenode.get('{%s}name' % (CNSD['style'],))
        tablename = name.split('.')[0]
        family = stylenode.get('{%s}family' % (CNSD['style'],))
        if name.startswith(TABLESTYLEPREFIX):
            tablestyle = table_styles.get(tablename)
            if tablestyle is None:
                tablestyle = TableStyle()
                table_styles[tablename] = tablestyle
            if family == 'table':
                properties = stylenode.find('{%s}table-properties' % (CNSD['style'],))
                property = properties.get('{%s}%s' % (CNSD['fo'], 'background-color'))
                if property is not None and property != 'none':
                    tablestyle.backgroundcolor = property
            elif family == 'table-cell':
                properties = stylenode.find('{%s}table-cell-properties' % (CNSD['style'],))
                if properties is not None:
                    border = self.get_property(properties)
                    if border is not None:
                        tablestyle.border = border
    return table_styles