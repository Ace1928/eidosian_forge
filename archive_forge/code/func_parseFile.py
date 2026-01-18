import sys
def parseFile(self, filename):
    data = open(filename, 'r').readlines()
    for line in data:
        self.readLine(line[:-1])
    self.endPara()
    return self._results