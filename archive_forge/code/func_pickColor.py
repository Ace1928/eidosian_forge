import string
from twisted.logger import Logger
def pickColor(self, value, mode, BOLD=ColorText.BOLD_COLORS):
    if mode:
        return ColorText.COLORS[value]
    else:
        return self.bold and BOLD[value] or ColorText.COLORS[value]