import _markupbase
import re
def report_unbalanced(self, tag):
    if self.verbose:
        print('*** Unbalanced </' + tag + '>')
        print('*** Stack:', self.stack)