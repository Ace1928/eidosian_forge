from suds import *
import suds.metrics as metrics
from suds.sax import Namespace
from logging import getLogger
def paramtypes(self):
    """Get all parameter types."""
    for m in [p[1] for p in self.ports]:
        for p in [p[1] for p in m]:
            for pd in p:
                if pd[1] in self.params:
                    continue
                item = (pd[1], pd[1].resolve())
                self.params.append(item)