import re
import itertools
@staticmethod
def getRsBlockTable(version, errorCorrectLevel):
    if errorCorrectLevel == QRErrorCorrectLevel.L:
        return QRRSBlock.RS_BLOCK_TABLE[(version - 1) * 4 + 0]
    elif errorCorrectLevel == QRErrorCorrectLevel.M:
        return QRRSBlock.RS_BLOCK_TABLE[(version - 1) * 4 + 1]
    elif errorCorrectLevel == QRErrorCorrectLevel.Q:
        return QRRSBlock.RS_BLOCK_TABLE[(version - 1) * 4 + 2]
    elif errorCorrectLevel == QRErrorCorrectLevel.H:
        return QRRSBlock.RS_BLOCK_TABLE[(version - 1) * 4 + 3]
    else:
        return None