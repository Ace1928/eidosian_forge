from prov.model import *
def test_entity_9(self):
    document = self.new_document()
    a = document.entity(EX_NS['e9'])
    a.add_attributes([(PROV_LABEL, 'entity9')])
    self.add_types(a)
    self.add_locations(a)
    self.add_labels(a)
    self.add_further_attributes(a)
    self.do_tests(document)