from prov.model import *
def test_entity_2(self):
    document = self.new_document()
    a = document.entity(EX_NS['e2'])
    a.add_attributes([(PROV_LABEL, 'entity2')])
    self.do_tests(document)