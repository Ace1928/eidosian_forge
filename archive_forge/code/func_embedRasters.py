from __future__ import division         # use "true" division instead of integer division in Python 2 (see PEP 238)
from __future__ import print_function   # use print() as a function in Python 2 (see PEP 3105)
from __future__ import absolute_import  # use absolute imports by default in Python 2 (see PEP 328)
import math
import optparse
import os
import re
import sys
import time
import xml.dom.minidom
from xml.dom import Node, NotFoundErr
from collections import namedtuple, defaultdict
from decimal import Context, Decimal, InvalidOperation, getcontext
import six
from six.moves import range, urllib
from scour.svg_regex import svg_parser
from scour.svg_transform import svg_transform_parser
from scour.yocto_css import parseCssString
from scour import __version__
def embedRasters(element, options):
    import base64
    '\n      Converts raster references to inline images.\n      NOTE: there are size limits to base64-encoding handling in browsers\n    '
    global _num_rasters_embedded
    href = element.getAttributeNS(NS['XLINK'], 'href')
    if href != '' and len(href) > 1:
        ext = os.path.splitext(os.path.basename(href))[1].lower()[1:]
        if ext in ['png', 'jpg', 'gif']:
            href_fixed = href.replace('\\', '/')
            href_fixed = re.sub('file:/+', 'file:///', href_fixed)
            parsed_href = urllib.parse.urlparse(href_fixed)
            if parsed_href.scheme == '':
                parsed_href = parsed_href._replace(scheme='file')
                if href_fixed[0] == '/':
                    href_fixed = 'file://' + href_fixed
                else:
                    href_fixed = 'file:' + href_fixed
            working_dir_old = None
            if parsed_href.scheme == 'file' and parsed_href.path[0] != '/':
                if options.infilename:
                    working_dir_old = os.getcwd()
                    working_dir_new = os.path.abspath(os.path.dirname(options.infilename))
                    os.chdir(working_dir_new)
            try:
                file = urllib.request.urlopen(href_fixed)
                rasterdata = file.read()
                file.close()
            except Exception as e:
                print("WARNING: Could not open file '" + href + "' for embedding. The raster image will be kept as a reference but might be invalid. (Exception details: " + str(e) + ')', file=options.ensure_value('stdout', sys.stdout))
                rasterdata = ''
            finally:
                if working_dir_old is not None:
                    os.chdir(working_dir_old)
            if rasterdata != '':
                b64eRaster = base64.b64encode(rasterdata)
                if b64eRaster != '':
                    if ext == 'jpg':
                        ext = 'jpeg'
                    element.setAttributeNS(NS['XLINK'], 'href', 'data:image/' + ext + ';base64,' + b64eRaster.decode())
                    _num_rasters_embedded += 1
                    del b64eRaster