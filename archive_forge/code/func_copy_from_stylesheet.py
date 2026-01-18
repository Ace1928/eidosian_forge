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
def copy_from_stylesheet(self, outzipfile):
    """Copy images, settings, etc from the stylesheet doc into target doc.
        """
    stylespath = self.settings.stylesheet
    inzipfile = zipfile.ZipFile(stylespath, 'r')
    s1 = inzipfile.read('settings.xml')
    self.write_zip_str(outzipfile, 'settings.xml', s1)
    namelist = inzipfile.namelist()
    for name in namelist:
        if name.startswith('Pictures/'):
            imageobj = inzipfile.read(name)
            outzipfile.writestr(name, imageobj)
    inzipfile.close()