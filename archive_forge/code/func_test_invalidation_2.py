from prov.model import *
def test_invalidation_2(self):
    document = self.new_document()
    document.invalidation(EX_NS['e1'], identifier=EX_NS['inv2'], activity=EX_NS['a1'])
    self.do_tests(document)