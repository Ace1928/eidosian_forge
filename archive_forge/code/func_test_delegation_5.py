from prov.model import *
def test_delegation_5(self):
    document = self.new_document()
    document.delegation(EX_NS['e1'], EX_NS['ag1'])
    self.do_tests(document)