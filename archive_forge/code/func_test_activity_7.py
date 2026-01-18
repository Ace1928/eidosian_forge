from prov.model import *
def test_activity_7(self):
    document = self.new_document()
    a = document.activity(EX_NS['a7'])
    a.add_attributes([(PROV_LABEL, 'activity7')])
    self.add_types(a)
    self.add_locations(a)
    self.add_labels(a)
    self.do_tests(document)