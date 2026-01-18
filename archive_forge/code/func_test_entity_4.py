from prov.model import *
def test_entity_4(self):
    document = self.new_document()
    a = document.entity(EX_NS['e4'])
    a.add_attributes([(PROV_LABEL, 'entity4')])
    self.add_labels(a)
    self.do_tests(document)