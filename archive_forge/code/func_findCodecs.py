from PySide2 import QtCore, QtGui, QtWidgets
def findCodecs(self):
    codecMap = []
    iso8859RegExp = QtCore.QRegExp('ISO[- ]8859-([0-9]+).*')
    for mib in QtCore.QTextCodec.availableMibs():
        codec = QtCore.QTextCodec.codecForMib(mib)
        sortKey = codec_name(codec).upper()
        rank = 0
        if sortKey.startswith('UTF-8'):
            rank = 1
        elif sortKey.startswith('UTF-16'):
            rank = 2
        elif iso8859RegExp.exactMatch(sortKey):
            if len(iso8859RegExp.cap(1)) == 1:
                rank = 3
            else:
                rank = 4
        else:
            rank = 5
        codecMap.append((str(rank) + sortKey, codec))
    codecMap.sort()
    self.codecs = [item[-1] for item in codecMap]