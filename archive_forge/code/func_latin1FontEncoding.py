import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
def latin1FontEncoding(fontname):
    """use this to generating PS code for re-encoding a font as ISOLatin1
    from font with name 'fontname' defines reencoded font, 'fontname-ISOLatin1'"""
    latin1FontTemplate = '/%s findfont\ndup length dict begin\n  {1 index /FID ne\n        {def}\n        {pop pop}\n      ifelse\n   } forall\n   /Encoding ISOLatin1Encoding  def\n   currentdict\nend\n/%s-ISOLatin1 exch definefont pop\n'
    return latin1FontTemplate % (fontname, fontname)