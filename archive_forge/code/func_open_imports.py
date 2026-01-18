from suds import *
from suds.xsd import *
from suds.xsd.depsort import dependency_sort
from suds.xsd.sxbuiltin import *
from suds.xsd.sxbase import SchemaObject
from suds.xsd.sxbasic import Factory as BasicFactory
from suds.xsd.sxbuiltin import Factory as BuiltinFactory
from suds.sax import splitPrefix, Namespace
from suds.sax.element import Element
from logging import getLogger
def open_imports(self, options, loaded_schemata):
    """
        Instruct all contained L{sxbasic.Import} children to import all of
        their referenced schemas. The imported schema contents are I{merged}
        in.

        @param options: An options dictionary.
        @type options: L{options.Options}
        @param loaded_schemata: Already loaded schemata cache (URL --> Schema).
        @type loaded_schemata: dict

        """
    for imp in self.imports:
        imported = imp.open(options, loaded_schemata)
        if imported is None:
            continue
        imported.open_imports(options, loaded_schemata)
        log.debug('imported:\n%s', imported)
        self.merge(imported)