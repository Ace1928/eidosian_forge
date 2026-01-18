import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
def testStringWidthAlgorithms():
    rawdata = open('../../rlextra/rml2pdf/doc/rml_user_guide.prep').read()
    print('rawdata length %d' % len(rawdata))
    print('test one huge string...')
    test3widths([rawdata])
    print()
    words = rawdata.split()
    print('test %d shorter strings (average length %0.2f chars)...' % (len(words), 1.0 * len(rawdata) / len(words)))
    test3widths(words)