from ._base import *
@property
def model_exists(self):
    return self.model_ckpt and File.exists(self.model_ckpt)