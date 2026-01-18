from prov.model import *
def test_derivation_1(self):
    document = self.new_document()
    document.derivation(None, usedEntity=EX_NS['e1'], identifier=EX_NS['der1'])
    self.do_tests(document)