from collections import namedtuple
import warnings
def getparams(self):
    return _sunau_params(self.getnchannels(), self.getsampwidth(), self.getframerate(), self.getnframes(), self.getcomptype(), self.getcompname())