from prov.model import *
def test_bundle_3(self):
    document = self.new_document()
    bundle1 = ProvBundle(identifier=EX_NS['bundle1'])
    bundle1.usage(activity=EX_NS['a1'], entity=EX_NS['e1'], identifier=EX_NS['use1'])
    bundle1.entity(identifier=EX_NS['e1'])
    bundle1.activity(identifier=EX_NS['a1'])
    bundle2 = ProvBundle(identifier=EX_NS['bundle2'])
    bundle2.usage(activity=EX_NS['aa1'], entity=EX_NS['ee1'], identifier=EX_NS['use2'])
    bundle2.entity(identifier=EX_NS['ee1'])
    bundle2.activity(identifier=EX_NS['aa1'])
    document.add_bundle(bundle1)
    document.add_bundle(bundle2)
    self.do_tests(document)