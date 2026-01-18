from prov.model import *
def test_derivation_2(self):
    document = self.new_document()
    document.derivation(EX_NS['e2'], usedEntity=None, identifier=EX_NS['der2'])
    self.do_tests(document)