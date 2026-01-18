from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def buildContext(stylesheet=None):
    result = {}
    from reportlab.lib.styles import getSampleStyleSheet
    if stylesheet is not None:
        for stylenamekey, stylenamevalue in DEFAULT_ALIASES.items():
            if stylenamekey in stylesheet:
                result[stylenamekey] = stylesheet[stylenamekey]
        for stylenamekey, stylenamevalue in DEFAULT_ALIASES.items():
            if stylenamevalue in stylesheet:
                result[stylenamekey] = stylesheet[stylenamevalue]
    styles = getSampleStyleSheet()
    for stylenamekey, stylenamevalue in DEFAULT_ALIASES.items():
        if stylenamekey not in result and stylenamevalue in styles:
            result[stylenamekey] = styles[stylenamevalue]
    return result