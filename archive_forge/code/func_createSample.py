import sys, time
from reportlab import Version as __RL_Version__
from reportlab.graphics.barcode.common import *
from reportlab.graphics.barcode.code39 import *
from reportlab.graphics.barcode.code93 import *
from reportlab.graphics.barcode.code128 import *
from reportlab.graphics.barcode.usps import *
from reportlab.graphics.barcode.usps4s import USPS_4State
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.barcode.dmtx import DataMatrixWidget, pylibdmtx
from reportlab.platypus import Spacer, SimpleDocTemplate, PageBreak
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.flowables import XBox, KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, Line
from reportlab.graphics.barcode import getCodeNames, createBarcodeDrawing, createBarcodeImageInMemory
def createSample(name, memory):
    f = open(name, 'wb')
    f.write(memory)
    f.close()