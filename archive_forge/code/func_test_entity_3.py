from prov.model import *
def test_entity_3(self):
    document = self.new_document()
    a = document.entity(EX_NS['e3'])
    a.add_attributes([(PROV_LABEL, 'entity3')])
    self.add_value(a)
    self.do_tests(document)