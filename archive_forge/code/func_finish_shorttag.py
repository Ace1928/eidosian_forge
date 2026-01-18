import _markupbase
import re
def finish_shorttag(self, tag, data):
    self.finish_starttag(tag, [])
    self.handle_data(data)
    self.finish_endtag(tag)