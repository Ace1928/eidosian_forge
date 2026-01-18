from prov.model import *
def test_delegation_1(self):
    document = self.new_document()
    document.delegation(EX_NS['e1'], None, identifier=EX_NS['dele1'])
    self.do_tests(document)