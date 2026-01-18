from prov.model import *
def test_activity_2(self):
    document = self.new_document()
    a = document.activity(EX_NS['a2'])
    a.add_attributes([(PROV_LABEL, 'activity2')])
    self.do_tests(document)