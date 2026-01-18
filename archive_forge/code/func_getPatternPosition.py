import re
import itertools
@staticmethod
def getPatternPosition(version):
    return QRUtil.PATTERN_POSITION_TABLE[version - 1]