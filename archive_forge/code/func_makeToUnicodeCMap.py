from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def makeToUnicodeCMap(fontname, subset):
    """Creates a ToUnicode CMap for a given subset.  See Adobe
    _PDF_Reference (ISBN 0-201-75839-3) for more information."""
    cmap = ['/CIDInit /ProcSet findresource begin', '12 dict begin', 'begincmap', '/CIDSystemInfo', '<< /Registry (%s)' % fontname, '/Ordering (%s)' % fontname, '/Supplement 0', '>> def', '/CMapName /%s def' % fontname, '/CMapType 2 def', '1 begincodespacerange', '<00> <%02X>' % (len(subset) - 1), 'endcodespacerange', '%d beginbfchar' % len(subset)] + ['<%02X> <%04X>' % (i, v) for i, v in enumerate(subset)] + ['endbfchar', 'endcmap', 'CMapName currentdict /CMap defineresource pop', 'end', 'end']
    return '\n'.join(cmap)