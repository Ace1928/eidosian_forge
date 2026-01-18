from prov.model import *
def test_entity_5(self):
    document = self.new_document()
    a = document.entity(EX_NS['e5'])
    a.add_attributes([(PROV_LABEL, 'entity5')])
    self.add_types(a)
    self.do_tests(document)