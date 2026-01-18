from prov.model import *
def test_activity_5(self):
    document = self.new_document()
    a = document.activity(EX_NS['a5'])
    a.add_attributes([(PROV_LABEL, 'activity5')])
    self.add_types(a)
    self.do_tests(document)