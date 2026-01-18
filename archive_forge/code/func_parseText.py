import sys
def parseText(self, textBlock):
    """Parses the a possible multi-line text block"""
    lines = textBlock.split('\n')
    for line in lines:
        self.readLine(line)
    self.endPara()
    return self._results