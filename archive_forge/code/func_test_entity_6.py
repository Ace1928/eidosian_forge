from prov.model import *
def test_entity_6(self):
    document = self.new_document()
    a = document.entity(EX_NS['e6'])
    a.add_attributes([(PROV_LABEL, 'entity6')])
    self.add_locations(a)
    self.do_tests(document)