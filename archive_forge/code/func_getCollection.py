import os
from twisted.spread import pb
def getCollection(self, name, domain, password, callback, errback):
    requestID = self.newRequestID()
    self.waitingForAnswers[requestID] = (callback, errback)
    self.sendCall('getCollection', requestID, name, domain, password)