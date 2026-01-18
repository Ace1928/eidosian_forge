from manilaclient.tests.unit import utils
def test_import_v1_share_type_access_module(self):
    try:
        from manilaclient.v1 import share_type_access
    except Exception as e:
        msg = "module 'manilaclient.v1.share_type_access' cannot be imported with error: %s" % str(e)
        assert False, msg
    for cls in ('ShareTypeAccess', 'ShareTypeAccessManager'):
        msg = "Module 'share_type_access' has no '%s' attr." % cls
        self.assertTrue(hasattr(share_type_access, cls), msg)