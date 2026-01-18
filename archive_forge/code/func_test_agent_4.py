from prov.model import *
def test_agent_4(self):
    document = self.new_document()
    a = document.agent(EX_NS['ag4'])
    a.add_attributes([(PROV_LABEL, 'agent4'), (PROV_LABEL, Literal('hello')), (PROV_LABEL, Literal('bye', langtag='en'))])
    self.do_tests(document)