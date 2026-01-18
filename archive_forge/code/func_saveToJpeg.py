from sping import colors
from sping.TK import TKCanvas, TKCanvasPIL
from Tkinter import *
def saveToJpeg(self):
    print('Saving canvas to file tkCanvasPIL.jpg')
    self.tkpil.save(file='tkCanvasPIL.jpg')