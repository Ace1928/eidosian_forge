from manilaclient.tests.unit import utils
def test_import_v1_share_servers_module(self):
    try:
        from manilaclient.v1 import share_servers
    except Exception as e:
        msg = "module 'manilaclient.v1.share_servers' cannot be imported with error: %s" % str(e)
        assert False, msg
    for cls in ('ShareServer', 'ShareServerManager'):
        msg = "Module 'share_servers' has no '%s' attr." % cls
        self.assertTrue(hasattr(share_servers, cls), msg)