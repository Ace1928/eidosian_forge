import re
import itertools
def calculate_version(self):
    for version in range(1, 40):
        rsBlocks = QRRSBlock.getRSBlocks(version, self.errorCorrectLevel)
        totalDataCount = sum((block.dataCount for block in rsBlocks))
        length = 0
        for data in self.dataList:
            length += 4
            length += data.getLengthBits(version)
            length += data.bitlength
        if length <= totalDataCount * 8:
            break
    return version