from prov.model import *
def test_agent_2(self):
    document = self.new_document()
    a = document.agent(EX_NS['ag2'])
    a.add_attributes([(PROV_LABEL, 'agent2')])
    self.do_tests(document)