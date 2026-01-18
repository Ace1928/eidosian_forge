from prov.model import *
def test_association_8(self):
    document = self.new_document()
    assoc = document.association(EX_NS['a1'], agent=EX_NS['ag1'], identifier=EX_NS['assoc8'], plan=EX_NS['plan1'])
    assoc.add_attributes([(PROV_ROLE, 'figroll'), (PROV_ROLE, 'sausageroll')])
    self.do_tests(document)