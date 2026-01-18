import os
def stringwidth(self, text, font):
    widths = self.getfont(font.lower())
    w = 0
    for char in text:
        w = w + widths[ord(char)]
    return w