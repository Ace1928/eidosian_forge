from prov.model import *
def test_invalidation_1(self):
    document = self.new_document()
    document.invalidation(EX_NS['e1'], identifier=EX_NS['inv1'])
    self.do_tests(document)