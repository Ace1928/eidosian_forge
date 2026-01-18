from rdkit.sping.colors import *
def drawMultiLineString(self, s, x, y, font=None, color=None, angle=0, **kwargs):
    """Breaks string into lines (on 
, \r, 
\r, or \r
), and calls drawString on each."""
    import math
    h = self.fontHeight(font)
    dy = h * math.cos(angle * math.pi / 180.0)
    dx = h * math.sin(angle * math.pi / 180.0)
    s = s.replace('\r\n', '\n')
    s = s.replace('\n\r', '\n')
    s = s.replace('\r', '\n')
    lines = s.split('\n')
    for line in lines:
        self.drawString(line, x, y, font, color, angle)
        x = x + dx
        y = y + dy