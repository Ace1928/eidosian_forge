from prov.model import *
def test_entity_10(self):
    document = self.new_document()
    a = document.entity(EX_NS['e10'])
    a.add_attributes([(PROV_LABEL, 'entity10')])
    self.add_types(a)
    self.add_locations(a)
    self.add_labels(a)
    self.add_further_attributes0(a)
    self.do_tests(document)