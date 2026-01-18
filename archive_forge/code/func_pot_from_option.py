import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def pot_from_option(self, opt, context=None, note='test'):
    sio = StringIO()
    exporter = export_pot._PotExporter(sio)
    if context is None:
        context = export_pot._ModuleContext('nowhere', 0)
    export_pot._write_option(exporter, context, opt, note)
    return sio.getvalue()