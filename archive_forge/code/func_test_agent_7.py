from prov.model import *
def test_agent_7(self):
    document = self.new_document()
    a = document.agent(EX_NS['ag7'])
    a.add_attributes([(PROV_LABEL, 'agent7')])
    self.add_locations(a)
    self.add_labels(a)
    self.do_tests(document)