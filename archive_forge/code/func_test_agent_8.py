from prov.model import *
def test_agent_8(self):
    document = self.new_document()
    a = document.agent(EX_NS['ag8'])
    a.add_attributes([(PROV_LABEL, 'agent8')])
    self.add_types(a)
    self.add_locations(a)
    self.add_labels(a)
    self.add_further_attributes(a)
    self.do_tests(document)