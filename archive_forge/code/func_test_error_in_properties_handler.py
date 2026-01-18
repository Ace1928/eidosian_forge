import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_error_in_properties_handler(self):
    """Log includes the custom properties returned by the registered
        handlers.
        """
    wt = self.make_standard_commit('error_in_properties_handler', revprops={'first_prop': 'first_value'})
    sio = self.make_utf8_encoded_stringio()
    formatter = log.LongLogFormatter(to_file=sio)

    def trivial_custom_prop_handler(revision):
        raise Exception('a test error')
    log.properties_handler_registry.register('trivial_custom_prop_handler', trivial_custom_prop_handler)
    log.show_log(wt.branch, formatter)
    self.assertContainsRe(sio.getvalue(), b'brz: ERROR: Exception: a test error')