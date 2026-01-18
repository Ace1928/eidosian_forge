import time, calendar
def getdomainliteral(self):
    """Parse an RFC 2822 domain-literal."""
    return '[%s]' % self.getdelimited('[', ']\r', False)