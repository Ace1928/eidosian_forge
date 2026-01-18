import py
import sys
def getconsumer(self, keywords):
    """ return a consumer matching the given keywords.

            tries to find the most suitable consumer by walking, starting from
            the back, the list of keywords, the first consumer matching a
            keyword is returned (falling back to py.log.default)
        """
    for i in range(len(keywords), 0, -1):
        try:
            return self.keywords2consumer[keywords[:i]]
        except KeyError:
            continue
    return self.keywords2consumer.get('default', default_consumer)