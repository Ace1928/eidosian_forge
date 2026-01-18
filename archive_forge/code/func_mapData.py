import re
import itertools
def mapData(self, data, maskPattern):
    bits = self.dataBitIterator(data)
    mask = QRUtil.getMask(maskPattern)
    for (col, row), dark in zip_longest(self.dataPosIterator(), bits, fillvalue=False):
        self.modules[row][col] = dark ^ mask(row, col)