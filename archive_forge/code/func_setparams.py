from collections import namedtuple
import warnings
def setparams(self, params):
    nchannels, sampwidth, framerate, nframes, comptype, compname = params
    self.setnchannels(nchannels)
    self.setsampwidth(sampwidth)
    self.setframerate(framerate)
    self.setnframes(nframes)
    self.setcomptype(comptype, compname)