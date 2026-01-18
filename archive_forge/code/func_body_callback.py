import sys
import pycurl
def body_callback(self, buf):
    self.contents = self.contents + buf