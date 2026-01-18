import logging
def start_p(self, attrs=None):
    if self.do_p:
        self.doc.write('\n\n%s' % self.spaces())