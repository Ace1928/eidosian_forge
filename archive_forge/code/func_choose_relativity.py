import struct
import dns.exception
import dns.rdata
import dns.name
def choose_relativity(self, origin=None, relativize=True):
    self.mname = self.mname.choose_relativity(origin, relativize)
    self.rname = self.rname.choose_relativity(origin, relativize)