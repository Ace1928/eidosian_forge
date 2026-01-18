import codecs
def setstate(self, state):
    codecs.BufferedIncrementalDecoder.setstate(self, state)
    self.first = state[1]