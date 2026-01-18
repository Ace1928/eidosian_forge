from prov.model import *
def test_derivation_11(self):
    document = self.new_document()
    document.revision(EX_NS['e2'], usedEntity=EX_NS['e1'], identifier=EX_NS['rev1'], activity=EX_NS['a'], usage=EX_NS['u'], generation=EX_NS['g'])
    self.do_tests(document)