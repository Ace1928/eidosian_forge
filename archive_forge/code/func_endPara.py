import sys
def endPara(self):
    text = ' '.join(self._buf)
    if text:
        if self._mode == PREFORMATTED:
            self._results.append(('PREFORMATTED', self._style, '\n'.join(self._buf)))
        else:
            self._results.append(('PARAGRAPH', self._style, text))
    self._buf = []
    self._style = 'Normal'