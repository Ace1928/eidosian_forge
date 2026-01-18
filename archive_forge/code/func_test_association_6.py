from prov.model import *
def test_association_6(self):
    document = self.new_document()
    assoc = document.association(EX_NS['a1'], agent=EX_NS['ag1'], identifier=EX_NS['assoc6'], plan=EX_NS['plan1'])
    self.add_labels(assoc)
    self.do_tests(document)