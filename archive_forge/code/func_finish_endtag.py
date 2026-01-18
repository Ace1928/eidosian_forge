import _markupbase
import re
def finish_endtag(self, tag):
    if not tag:
        found = len(self.stack) - 1
        if found < 0:
            self.unknown_endtag(tag)
            return
    else:
        if tag not in self.stack:
            try:
                method = getattr(self, 'end_' + tag)
            except AttributeError:
                self.unknown_endtag(tag)
            else:
                self.report_unbalanced(tag)
            return
        found = len(self.stack)
        for i in range(found):
            if self.stack[i] == tag:
                found = i
    while len(self.stack) > found:
        tag = self.stack[-1]
        try:
            method = getattr(self, 'end_' + tag)
        except AttributeError:
            method = None
        if method:
            self.handle_endtag(tag, method)
        else:
            self.unknown_endtag(tag)
        del self.stack[-1]