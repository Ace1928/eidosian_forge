import time
from twisted.internet import defer
from twisted.spread import banana, flavors, jelly
def getFileName(self, ext='pub'):
    return '{}-{}-{}.{}'.format(self.service, self.perspective, str(self.publishedID), ext)