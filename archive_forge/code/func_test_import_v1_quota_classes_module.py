from manilaclient.tests.unit import utils
def test_import_v1_quota_classes_module(self):
    try:
        from manilaclient.v1 import quota_classes
    except Exception as e:
        msg = "module 'manilaclient.v1.quota_classes' cannot be imported with error: %s" % str(e)
        assert False, msg
    for cls in ('QuotaClassSet', 'QuotaClassSetManager'):
        msg = "Module 'quota_classes' has no '%s' attr." % cls
        self.assertTrue(hasattr(quota_classes, cls), msg)