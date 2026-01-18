from lxml import etree
import sys
import zipfile
from tempfile import mkstemp
import shutil
import os
def prepstyle(filename):
    zin = zipfile.ZipFile(filename)
    styles = zin.read('styles.xml')
    root = etree.fromstring(styles)
    for el in root.xpath('//style:page-layout-properties', namespaces=NAMESPACES):
        for attr in el.attrib:
            if attr.startswith('{%s}' % NAMESPACES['fo']):
                del el.attrib[attr]
    tempname = mkstemp()
    zout = zipfile.ZipFile(os.fdopen(tempname[0], 'w'), 'w', zipfile.ZIP_DEFLATED)
    for item in zin.infolist():
        if item.filename == 'styles.xml':
            zout.writestr(item, etree.tostring(root))
        else:
            zout.writestr(item, zin.read(item.filename))
    zout.close()
    zin.close()
    shutil.move(tempname[1], filename)