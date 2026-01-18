from pidWxDc import PiddleWxDc
from wxPython.wx import *
def ignoreClick(canvas, x, y):
    canvas.sb.OnClick(x, y)