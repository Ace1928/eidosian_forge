import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def math2html(formula):
    """Convert some TeX math to HTML."""
    factory = FormulaFactory()
    whole = factory.parseformula(formula)
    FormulaProcessor().process(whole)
    whole.process()
    return ''.join(whole.gethtml())