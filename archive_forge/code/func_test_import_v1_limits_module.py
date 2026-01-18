from manilaclient.tests.unit import utils
def test_import_v1_limits_module(self):
    try:
        from manilaclient.v1 import limits
    except Exception as e:
        msg = "module 'manilaclient.v1.limits' cannot be imported with error: %s" % str(e)
        assert False, msg
    for cls in ('Limits', 'RateLimit', 'AbsoluteLimit', 'LimitsManager'):
        msg = "Module 'limits' has no '%s' attr." % cls
        self.assertTrue(hasattr(limits, cls), msg)