from prov.model import *
def test_derivation_13(self):
    document = self.new_document()
    document.primary_source(EX_NS['e2'], usedEntity=EX_NS['e1'], identifier=EX_NS['prim1'], activity=EX_NS['a'], usage=EX_NS['u'], generation=EX_NS['g'])
    self.do_tests(document)