import formatter
import string
from types import *
import htmllib
import piddle
def send_label_data(self, data):
    if data == '*':
        w = self.pc.stringWidth(data, self.font) / 3
        h = self.pc.fontHeight(self.font) / 3
        x = self.indent - w
        y = self.y - w
        self.pc.drawRect(x, y, x - w, y - w)
    else:
        w = self.pc.stringWidth(data, self.font)
        h = self.pc.fontHeight(self.font)
        x = self.indent - w - self.fsizex / 3
        if x < 0:
            x = 0
        self.pc.drawString(data, x, self.y, self.font, self.color)