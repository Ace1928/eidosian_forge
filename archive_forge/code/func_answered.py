import sys, pycurl
def answered(self, check):
    """Did a given check string occur in the last payload?"""
    return self.payload.find(check) >= 0