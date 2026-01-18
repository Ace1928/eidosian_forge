from prov.model import *
def test_derivation_7(self):
    document = self.new_document()
    document.derivation(EX_NS['e2'], usedEntity=EX_NS['e1'], identifier=EX_NS['der7'], activity=EX_NS['a'], usage=EX_NS['u'], generation=EX_NS['g'])
    self.do_tests(document)