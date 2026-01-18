from prov.model import *
def test_membership_3(self):
    document = self.new_document()
    document.membership(EX_NS['c'], EX_NS['e1'])
    document.membership(EX_NS['c'], EX_NS['e2'])
    document.membership(EX_NS['c'], EX_NS['e3'])
    self.do_tests(document)