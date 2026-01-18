import formatter
import string
from types import *
import htmllib
import piddle
def send_literal_data(self, data):
    if not data:
        return
    lines = data.split(data, '\n')
    text = lines[0].replace('\t', ' ' * 8)
    for l in lines[1:]:
        self.OutputLine(text, 1)
        text = l.replace('\t', ' ' * 8)
    self.OutputLine(text, 0)
    self.atbreak = 0