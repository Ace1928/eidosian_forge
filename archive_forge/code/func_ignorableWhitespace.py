import xml.sax
import xml.sax.handler
def ignorableWhitespace(self, chars):
    PullDOM.ignorableWhitespace(self, chars)
    node = self.lastEvent[0][1]
    parentNode = self.elementStack[-1]
    parentNode.appendChild(node)