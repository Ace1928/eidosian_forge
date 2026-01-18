import formatter
import string
from types import *
import htmllib
import piddle
def send_flowing_data(self, data):
    if not data:
        return
    atbreak = self.atbreak or data[0] in string.whitespace
    text = ''
    pixels = chars = 0
    for word in data.split():
        bword = ' ' + word
        length = len(bword)
        if not atbreak:
            text = word
            chars = chars + length - 1
        elif self.x + pixels + (chars + length) * self.fsizex < self.rmargin:
            text = text + bword
            chars = chars + length
        else:
            w = self.pc.stringWidth(text + bword, self.font)
            h = self.pc.fontHeight(self.font)
            if TRACE:
                print('sfd T:', text + bword)
            if TRACE:
                print('sfd', self.x, w, self.x + w, self.rmargin)
            if self.x + w < self.rmargin:
                text = text + bword
                pixels = w
                chars = 0
            else:
                self.OutputLine(text, 1)
                text = word
                chars = length - 1
                pixels = 0
        atbreak = 1
    self.OutputLine(text, 0)
    self.atbreak = data[-1] in string.whitespace