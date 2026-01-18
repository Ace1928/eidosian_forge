from prov.model import *
def test_association_2(self):
    document = self.new_document()
    document.association(None, agent=EX_NS['ag1'], identifier=EX_NS['assoc2'])
    self.do_tests(document)