from prov.model import *
def test_association_4(self):
    document = self.new_document()
    document.association(EX_NS['a1'], agent=EX_NS['ag1'], identifier=EX_NS['assoc4'], plan=EX_NS['plan1'])
    self.do_tests(document)