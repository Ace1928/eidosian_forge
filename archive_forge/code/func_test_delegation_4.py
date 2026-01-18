from prov.model import *
def test_delegation_4(self):
    document = self.new_document()
    document.delegation(EX_NS['e1'], EX_NS['ag1'], activity=EX_NS['a1'], identifier=EX_NS['dele4'])
    self.do_tests(document)