from .. import config, debug, tests
def assertDebugFlags(self, expected_flags, conf_bytes):
    conf = config.GlobalStack()
    conf.store._load_from_string(b'[DEFAULT]\n' + conf_bytes)
    conf.store.save()
    self.overrideAttr(debug, 'debug_flags', set())
    debug.set_debug_flags_from_config()
    self.assertEqual(set(expected_flags), debug.debug_flags)